#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time
import logging
import threading # Used for CLI timing if needed, SocketIO handles its own background tasks

# --- Flask and SocketIO Imports ---
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# --- Global Variables for Web UI ---
# These will be loaded once if running in web UI mode
app = Flask(__name__)
# Secret key is needed for session management, even if not explicitly used
app.config['SECRET_KEY'] = os.urandom(24)
# Use eventlet for async operations, suitable for long-polling and websockets
socketio = SocketIO(app, async_mode='eventlet')
# Global storage for loaded data to avoid reloading on each request
loaded_data = {
    "df": None,
    "nutrient_mapping": None,
    "rdi_values": None
}
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Logic Functions (Modified slightly for flexibility) ---

def get_next_run_number():
    """Get the next run number by checking existing CSV file"""
    history_file = 'optimization_history.csv'
    if not os.path.exists(history_file):
        return 1
    try:
        df = pd.read_csv(history_file)
        if not df.empty and 'run_number' in df.columns:
            return df['run_number'].max() + 1
        else:
            return 1
    except pd.errors.EmptyDataError:
        return 1
    except Exception as e:
        logging.error(f"Error reading optimization history: {e}")
        return 1 # Fallback

def _load_data(excel_path="Release 2 - Nutrient file.xlsx", rdi_json_path="rdi/rdi.json"):
    """Loads Excel and RDI data."""
    logging.info("Loading data...")
    try:
        df = pd.read_excel(excel_path, sheet_name="All solids & liquids per 100g")
        # Clean column names immediately after loading
        df.columns = [clean_column_name(col) for col in df.columns]
        logging.info(f"Loaded Excel data: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        logging.error(f"Error: Excel file not found at {excel_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading Excel file: {e}")
        return None, None, None

    try:
        with open(rdi_json_path, 'r', encoding='utf-8') as f:
            nutrient_mapping = json.load(f)
        rdi_values = {nutrient: details.get('rdi', 0) for nutrient, details in nutrient_mapping.items()}
        logging.info(f"Loaded RDI data for {len(rdi_values)} nutrients.")
    except FileNotFoundError:
        logging.error(f"Error: RDI JSON file not found at {rdi_json_path}")
        return df, None, None # Return df even if RDI fails, maybe user wants raw data
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {rdi_json_path}")
        return df, None, None
    except Exception as e:
        logging.error(f"Error loading RDI file: {e}")
        return df, None, None

    return df, nutrient_mapping, rdi_values

def clean_column_name(col_name):
    """Clean column names by removing extra whitespace and newlines"""
    if isinstance(col_name, str):
        return col_name.strip().replace('\n', '')
    return col_name # Return original if not a string

def optimize_nutrition(food_df,
                       nutrient_mapping,
                       rdi_targets,
                       number_of_meals=1,
                       meal_number=1, # Kept for potential future use, but current setup uses 1 meal
                       randomness_factor=0.3, # Currently unused, consider integrating if needed
                       population_size=50,
                       generations=100,
                       diet_type='all',
                       run_number=None, # Pass run_number explicitly
                       socketio_instance=None, # For web UI real-time updates
                       sid=None): # Session ID for targeting updates
    """
    Optimize food selection (Core logic). Emits progress via SocketIO if provided.
    """
    start_time = time.time() # Track time for this specific optimization run

    # Validate inputs
    if meal_number < 1 or meal_number > number_of_meals:
        raise ValueError(f"Meal number must be between 1 and {number_of_meals}")

    if food_df is None or food_df.empty:
        logging.error("Food DataFrame is empty or None. Cannot optimize.")
        if socketio_instance and sid:
             socketio_instance.emit('optimization_error', {'message': 'No food data available for optimization.'}, room=sid)
        return None # Indicate failure

    if not nutrient_mapping or not rdi_targets:
        logging.error("Nutrient mapping or RDI targets are missing.")
        if socketio_instance and sid:
             socketio_instance.emit('optimization_error', {'message': 'Nutrient mapping or RDI targets missing.'}, room=sid)
        return None

    logging.info(f"Starting optimization run #{run_number} for diet '{diet_type}' with {generations} generations, {population_size} population.")

    # Ensure RDI targets only include nutrients present in mapping
    valid_rdi_targets = {k: v for k, v in rdi_targets.items() if k in nutrient_mapping}
    if len(valid_rdi_targets) != len(rdi_targets):
        logging.warning("Some RDI targets were removed as they were not in the nutrient mapping.")
    if not valid_rdi_targets:
        logging.error("No valid RDI targets found after checking against nutrient mapping.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'No valid RDI targets found.'}, room=sid)
        return None

    # Scale RDI targets for a single meal (even if number_of_meals is 1)
    meal_rdi_targets = {nutrient: target / number_of_meals for nutrient, target in valid_rdi_targets.items() if target is not None}

    # --- Prepare food data ---
    available_foods = {}
    nutrient_cols_in_df = [col for col in nutrient_mapping.keys() if col in food_df.columns]
    if not nutrient_cols_in_df:
         logging.error("None of the nutrients in the mapping are present as columns in the provided food_df.")
         if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'Mismatch between nutrient mapping and food data columns.'}, room=sid)
         return None

    for idx, row in food_df.iterrows():
        food_name = row['Food Name']
        # Ensure food_name is hashable (string)
        if not isinstance(food_name, str):
             logging.warning(f"Skipping food with non-string name at index {idx}: {food_name}")
             continue

        food_data = {'density': 100} # Assume 100g base unit
        for nutrient_col in nutrient_cols_in_df:
            food_data[nutrient_col] = pd.to_numeric(row[nutrient_col], errors='coerce') # Convert to numeric, force errors to NaN
            if pd.isna(food_data[nutrient_col]):
                food_data[nutrient_col] = 0.0 # Replace NaN with 0

        available_foods[food_name] = food_data

    if not available_foods:
        logging.error("No valid foods could be processed from the input DataFrame.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'No valid foods found in the selected data.'}, room=sid)
        return None

    # Convert to DataFrame for easier manipulation (only necessary if calculations need it)
    # foods_df = pd.DataFrame(available_foods).T # Keep if nutrient_score calculation is used

    # --- Genetic Algorithm Setup ---
    def create_solution():
        """Create a random solution (diet plan)."""
        # Ensure amount is non-negative
        return {food: max(0, random.uniform(10, 150)) for food in available_foods} # Start with reasonable amounts

    # Generate random penalties for this optimization run
    penalties = {
        "under_rdi": random.uniform(1.5, 3.0), # Penalize missing nutrients more
        "over_rdi": random.uniform(0.1, 0.5),  # Less penalty for general overage
        "over_ul": random.uniform(1.0, 2.5),   # Moderate penalty for exceeding UL (if UL data available)
        "water_soluble": random.uniform(0.05, 0.2), # Very lenient for excess water-soluble
        "fat_soluble": random.uniform(0.2, 0.6)    # More lenient for fat-soluble than general overage
    }
    logging.info(f"Run #{run_number} Penalties: {penalties}")
    if socketio_instance and sid:
        socketio_instance.emit('status_update', {'message': f"Generated penalties: {penalties}"}, room=sid)


    def evaluate_solution(solution):
        """Evaluate how close the solution is to the RDI targets."""
        current_nutrients = {nutrient: 0 for nutrient in meal_rdi_targets}
        for food, amount in solution.items():
            if food not in available_foods: continue # Safety check
            # Amount must be non-negative
            safe_amount = max(0, amount)
            food_details = available_foods[food]
            for nutrient in meal_rdi_targets:
                 # Use .get() for safety in case a nutrient is somehow missing from food_details
                nutrient_val = food_details.get(nutrient, 0.0)
                # Ensure density is not zero to avoid division error
                density = food_details.get('density', 100.0)
                if density > 0:
                    nutrient_per_gram = nutrient_val / density
                    current_nutrients[nutrient] += nutrient_per_gram * safe_amount
        return _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties, nutrient_mapping) # Pass nutrient_mapping


    def mutate_solution(solution):
        """Mutate the solution by tweaking food amounts."""
        if not solution: return solution # Handle empty solution
        food_to_mutate = random.choice(list(solution.keys()))
        # Mutate by a percentage or fixed amount, ensure non-negative
        change_factor = random.uniform(-0.3, 0.3) # Change by up to 30%
        solution[food_to_mutate] = max(0, solution[food_to_mutate] * (1 + change_factor) + random.uniform(-10, 10))
        # Occasionally add/remove a food? (More complex mutation)
        return solution

    def crossover_solution(sol1, sol2):
        """Create a new solution by combining two parent solutions."""
        new_solution = {}
        all_foods = list(set(sol1.keys()) | set(sol2.keys()))
        for food in all_foods:
            # Average amounts or pick from one parent
            if random.random() < 0.5:
                 new_solution[food] = sol1.get(food, 0) # Use get with default 0
            else:
                 new_solution[food] = sol2.get(food, 0)
            # Could also average: new_solution[food] = (sol1.get(food, 0) + sol2.get(food, 0)) / 2
        return new_solution

    # --- Genetic Algorithm Execution ---
    population = [create_solution() for _ in range(population_size)]
    best_overall_solution = None
    best_overall_score = float('inf')
    history_data = [] # Collect history for CSV saving

    for generation in range(generations):
        # Evaluate population
        scores = []
        for sol in population:
            score = evaluate_solution(sol)
            scores.append((score, sol))

        if not scores:
            logging.error(f"Run #{run_number} Generation {generation+1}: No valid scores generated. Stopping.")
            if socketio_instance and sid:
                 socketio_instance.emit('optimization_error', {'message': 'Evaluation failed to produce scores.'}, room=sid)
            break # Stop if evaluation fails

        scores.sort(key=lambda x: x[0])
        current_best_score, current_best_solution = scores[0]

        if current_best_score < best_overall_score:
            best_overall_score = current_best_score
            best_overall_solution = current_best_solution

        # --- Real-time Update & History Logging ---
        timestamp_now = datetime.now()
        gen_info = {
            'run_number': run_number,
            'generation': generation + 1,
            'score': current_best_score,
            'timestamp': timestamp_now.strftime('%Y-%m-%d %H:%M:%S')
        }
        history_data.append(gen_info)

        # Emit progress to the specific web client, if applicable
        if socketio_instance and sid:
            socketio_instance.emit('generation_update', {
                'generation': generation + 1,
                'score': current_best_score,
                'total_generations': generations
            }, room=sid)
            # Give the server a tiny break to handle IO, prevents freezing on very fast loops
            socketio.sleep(0.01)

        # Log progress to console (useful for CLI mode or server logs)
        if (generation + 1) % 10 == 0 or generation == generations - 1:
             logging.info(f"Run #{run_number} - Gen {generation + 1}/{generations}, Best Score: {current_best_score:.4f}")


        # --- Selection, Crossover, Mutation ---
        # Select the best half (elitism)
        num_elites = max(1, population_size // 10) # Keep top 10%
        elites = [sol for _, sol in scores[:num_elites]]

        # Select parents for the rest based on score (e.g., tournament selection or roulette wheel)
        # Using simple truncation selection for now:
        selected_parents = [sol for _, sol in scores[:population_size // 2]]
        if not selected_parents: # Ensure we have parents if population size is small
             selected_parents = [sol for _, sol in scores]


        # Create new population
        new_population = elites[:] # Start with the elites
        while len(new_population) < population_size:
            # Ensure we have at least 2 parents to sample from
            if len(selected_parents) >= 2:
                parent1, parent2 = random.sample(selected_parents, 2)
            elif len(selected_parents) == 1:
                 parent1 = parent2 = selected_parents[0]
            else: # Should not happen if scores exist, but safety first
                 parent1 = parent2 = create_solution() # Fallback

            child = crossover_solution(parent1, parent2)
            if random.random() < 0.2: # Mutation probability (e.g., 20%)
                child = mutate_solution(child)
            new_population.append(child)

        population = new_population

    # --- Save History ---
    if history_data:
        history_df = pd.DataFrame(history_data)
        history_file = 'optimization_history.csv'
        try:
            if os.path.exists(history_file):
                history_df.to_csv(history_file, mode='a', header=False, index=False)
            else:
                history_df.to_csv(history_file, index=False)
        except IOError as e:
            logging.error(f"Error saving optimization history to CSV: {e}")

    execution_time = time.time() - start_time
    logging.info(f"Optimization run #{run_number} finished. Execution time: {execution_time:.2f} seconds. Best Score: {best_overall_score:.4f}")

    if best_overall_solution is None:
        logging.error(f"Run #{run_number}: No best solution found.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'Optimization finished but no solution was found.'}, room=sid)
        return None

    report_paths = save_nutrition_report(
        best_overall_solution,
        available_foods, # Pass the processed dict
        valid_rdi_targets, # Pass the RDI targets used for this meal
        nutrient_mapping,  # <--- Pass nutrient_mapping here
        best_overall_score,
        run_number,
        number_of_meals,
        meal_number,
        generations,
        execution_time,
        diet_type,
        penalties
    )

    logging.info(f"Report saved: JSON={report_paths['json']}, HTML={report_paths['html']}")

    socketio_instance.emit('optimization_complete', {
        'message': f"Optimization complete! Score: {best_overall_score:.2f}",
        'report_html': os.path.basename(report_paths['html']),
        'report_json': os.path.basename(report_paths['json']),
        'run_number': int(run_number) # <-- Cast run_number to standard Python int
    }, room=sid)

    # Round the amounts for final output
    final_solution = {food: round(amount) for food, amount in best_overall_solution.items() if round(amount) > 5} # Filter very small amounts

    return {
        "solution": final_solution,
        "score": best_overall_score,
        "penalties": penalties,
        "report_paths": report_paths,
        "run_number": run_number
    }


def _is_nutrition_sufficient(current, targets, threshold=0.90):
    """Check if current nutrients are at least threshold percent of targets."""
    # This function seems unused in the provided core logic flow, but keeping it.
    for nutrient, target in targets.items():
        if target > 0 and current.get(nutrient, 0) < threshold * target:
            return False
    return True

def _get_nutrient_type(nutrient_name, nutrient_mapping):
    """Determine if a nutrient is water-soluble, fat-soluble, or other based on mapping."""
    details = nutrient_mapping.get(nutrient_name, {})
    if details.get('water_soluble', False):
        return 'water_soluble'
    elif details.get('fat_soluble', False):
        return 'fat_soluble'
    elif details.get('is_mineral', False): # Example, add flags to your rdi.json
         return 'mineral'
    else:
        return 'other' # Includes protein, carbs, fiber, etc.


def _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties, nutrient_mapping):
    """Calculate how far current nutrients are from targets with randomized penalties."""
    score = 0
    ul_targets = {nutrient: details.get('ul') for nutrient, details in nutrient_mapping.items()} # Load UL values

    for nutrient, target in meal_rdi_targets.items():
        if target is None or target <= 0: # Skip nutrients with no target or invalid target
            continue

        current_value = current_nutrients.get(nutrient, 0)
        nutrient_type = _get_nutrient_type(nutrient, nutrient_mapping)
        upper_limit = ul_targets.get(nutrient)

        # --- Penalty for being UNDER target ---
        if current_value < target:
            # Sigmoid-like penalty: harsher penalty when very low, tapers off near target
            deficit_ratio = (target - current_value) / target
            # Apply base 'under_rdi' penalty, maybe scaled by importance if needed
            score += (deficit_ratio ** 2) * penalties["under_rdi"]

        # --- Penalty for being OVER target ---
        else:
            overage_ratio = (current_value - target) / target

            # Check against Upper Limit (UL) if available
            over_ul_penalty = 0
            if upper_limit is not None and upper_limit > 0 and current_value > upper_limit:
                # Calculate how much over UL, relative to the range between RDI and UL
                 ul_overage_ratio = (current_value - upper_limit) / upper_limit if upper_limit > 0 else 0
                 # Apply UL penalty - potentially higher than general overage
                 over_ul_penalty = (ul_overage_ratio ** 2) * penalties["over_ul"]


            # Determine base penalty based on nutrient type
            if nutrient_type == 'water_soluble':
                base_overage_penalty = penalties["water_soluble"]
            elif nutrient_type == 'fat_soluble':
                base_overage_penalty = penalties["fat_soluble"]
            else: # Macronutrients, minerals etc.
                base_overage_penalty = penalties["over_rdi"]

            # Apply penalty for general overage (above RDI but potentially below UL)
            general_overage_penalty = (overage_ratio ** 2) * base_overage_penalty

            # Add the larger of the UL penalty or the general overage penalty
            # This prevents double-penalizing but ensures the UL penalty takes precedence if exceeded
            score += max(general_overage_penalty, over_ul_penalty)


    # --- Penalty for too many or too few ingredients? ---
    # num_ingredients = len([amt for amt in current_nutrients.values() if amt > 0.1]) # Count non-zero ingredients
    # if num_ingredients < 5: score += 5 # Penalize too few ingredients
    # if num_ingredients > 20: score += (num_ingredients - 20) * 0.1 # Penalize too many

    return score


def _get_unit(nutrient_name, nutrient_mapping):
    """Get the unit for a nutrient from the mapping."""
    return nutrient_mapping.get(nutrient_name, {}).get('unit', '')

def save_nutrition_report(foods_consumed, food_data_dict, rdi, nutrient_mapping, score, run_number,
                         number_of_meals=1, meal_number=1, generations=100,
                         execution_time=0, diet_type='all', penalties=None):
    """Save a detailed nutrition report as JSON and HTML."""
    report_timestamp = datetime.now()
    timestamp_str = report_timestamp.strftime('%Y%m%d_%H%M%S')
    report_date_str = report_timestamp.strftime('%Y-%m-%d')

    # Ensure directories exist
    init_directories()

    # Use daily RDI for daily % calculation, meal RDI for meal %
    daily_rdi = {nutrient: float(target) for nutrient, target in rdi.items() if target is not None}
    meal_rdi = {nutrient: target / number_of_meals for nutrient, target in daily_rdi.items()}

    # Calculate final nutrients
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}
    for food, amount in foods_consumed.items():
        if food not in food_data_dict: continue
        safe_amount = max(0, amount)
        food_details = food_data_dict[food]
        density = food_details.get('density', 100.0)
        if density <= 0: density = 100.0 # Avoid division by zero

        for nutrient in meal_rdi:
            nutrient_val = food_details.get(nutrient, 0.0)
            nutrient_per_gram = nutrient_val / density
            final_nutrients[nutrient] += nutrient_per_gram * safe_amount

    # --- Build Report Dictionary ---
    report = {
        "meal_info": {
            "run_number": int(run_number),
            "diet_type": str(diet_type),
            "target_percentage": float(100/number_of_meals),
            "optimization_score": float(score),
            "generations": int(generations),
            "number_of_foods": len(foods_consumed),
            "execution_time_seconds": float(execution_time),
            "date": report_date_str,
            "timestamp": timestamp_str,
            "penalties": {k: float(f"{v:.3f}") for k, v in penalties.items()} if penalties else {}
        },
        "food_quantities": {
            str(food): f"{float(amount):.1f}g"
            for food, amount in sorted(foods_consumed.items(), key=lambda item: item[1], reverse=True)
            if float(amount) > 0.1 # Only include foods with non-negligible amounts
        },
        "nutrition_profile": {},
        "summary": {}
    }

    # Populate Nutrition Profile
    nutrients_low = 0
    nutrients_ok = 0
    nutrients_high = 0
    for nutrient, amount in final_nutrients.items():
        meal_target = meal_rdi.get(nutrient)
        daily_target = daily_rdi.get(nutrient)
        unit = _get_unit(nutrient, nutrient_mapping) # Get unit from mapping

        meal_percentage = 0
        if meal_target is not None and meal_target > 0:
             meal_percentage = (amount / meal_target) * 100

        daily_percentage = 0
        if daily_target is not None and daily_target > 0:
             daily_percentage = (amount / daily_target) * 100

        # Determine status based on meal percentage
        status = "OK"
        if meal_percentage < 85: # Stricter lower bound
             status = "LOW"
             nutrients_low += 1
        elif meal_percentage > 150: # Check Upper Limit here if desired
             # Example: Check UL before marking HIGH
             # ul_value = nutrient_mapping.get(nutrient, {}).get('ul')
             # if ul_value and amount > ul_value / number_of_meals:
             #    status = "VERY HIGH (Check UL)" # More specific status
             # else:
             #    status = "HIGH"
             status = "HIGH"
             nutrients_high += 1
        else:
            nutrients_ok += 1


        report["nutrition_profile"][nutrient] = {
            "amount": f"{amount:.1f}{unit}",
            "meal_percentage": f"{meal_percentage:.1f}%",
            "daily_percentage": f"{daily_percentage:.1f}%",
            "status": status
        }

    report["summary"] = {
        "nutrients_at_good_levels": nutrients_ok,
        "nutrients_below_target": nutrients_low,
        "nutrients_above_target": nutrients_high,
        "total_nutrients_tracked": len(meal_rdi)
    }


    # --- Save JSON ---
    json_filename = os.path.join('recipes', 'json', f'meal_{run_number}_{timestamp_str}.json')
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Failed to save JSON report {json_filename}: {e}")
        json_filename = None # Indicate failure

    # --- Generate and Save HTML ---
    html_filename = os.path.join('recipes', 'html', f'meal_{run_number}_{timestamp_str}.html')
    html_content = generate_html_report(report) # Use helper function
    try:
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except IOError as e:
        logging.error(f"Failed to save HTML report {html_filename}: {e}")
        html_filename = None # Indicate failure

    return {"json": json_filename, "html": html_filename}


def generate_html_report(report_data):
    """Generates HTML content from the report dictionary."""
    meal_info = report_data['meal_info']
    food_quantities = report_data['food_quantities']
    nutrition_profile = report_data['nutrition_profile']
    summary = report_data['summary']
    penalties = meal_info.get('penalties', {})

    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meal {meal_info['run_number']} - Nutrition Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.5; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: #f9f9f9; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-bottom: 15px;}}
        h1 {{ text-align: center; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.95em; }}
        th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #e9ecef; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .status-low {{ color: #dc3545; font-weight: bold; }}
        .status-ok {{ color: #28a745; }}
        .status-high {{ color: #ffc107; font-weight: bold; }}
        .status-very-high-check-ul {{ color: #e83e8c; font-weight: bold; background-color: #f9d6e4; }} /* Example for UL warning */
        .summary-box, .info-box, .penalties-box {{ background-color: #fff; padding: 15px; border: 1px solid #eee; border-radius: 5px; margin-bottom: 20px; }}
        .info-box p, .summary-box p, .penalties-box li {{ margin: 5px 0; }}
        .info-box strong, .summary-box strong {{ color: #555; }}
        .penalties-box ul {{ list-style: none; padding: 0; }}
        .penalties-box li {{ border-bottom: 1px dashed #eee; padding-bottom: 3px; margin-bottom: 3px; }}
        .penalties-box li:last-child {{ border-bottom: none; }}
        @media (max-width: 600px) {{
            th, td {{ padding: 6px 8px; }}
            body {{ margin: 10px; }}
            .container {{ padding: 15px; }}
            h1 {{ font-size: 1.5em; }}
            h2 {{ font-size: 1.2em; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Meal {meal_info['run_number']} Nutrition Report</h1>

    <div class="info-box">
        <h2>Meal Information</h2>
        <p><strong>Run Number:</strong> {meal_info['run_number']}</p>
        <p><strong>Diet Type:</strong> {meal_info['diet_type']}</p>
        <p><strong>Date Generated:</strong> {meal_info['date']}</p>
        <p><strong>Target Nutrition:</strong> {meal_info['target_percentage']:.0f}% of Daily RDI</p>
        <p><strong>Optimization Score:</strong> {meal_info['optimization_score']:.3f} (Lower is better)</p>
        <p><strong>Generations:</strong> {meal_info['generations']}</p>
        <p><strong>Foods Used:</strong> {meal_info['number_of_foods']}</p>
        <p><strong>Execution Time:</strong> {meal_info['execution_time_seconds']:.2f} seconds</p>
    </div>
"""
    # Food Quantities Table
    html += """
    <h2>Food Quantities</h2>
    <table>
        <thead><tr><th>Food</th><th>Amount</th></tr></thead>
        <tbody>
"""
    if food_quantities:
        for food, amount in food_quantities.items():
             html += f"            <tr><td>{food}</td><td>{amount}</td></tr>\n"
    else:
         html += "            <tr><td colspan='2'>No foods listed in final solution.</td></tr>\n"
    html += """
        </tbody>
    </table>
"""
    # Nutrition Profile Table
    html += """
    <h2>Nutrition Profile</h2>
    <table>
        <thead>
            <tr>
                <th>Nutrient</th>
                <th>Amount</th>
                <th>% of Meal Target</th>
                <th>% of Daily RDI</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
    if nutrition_profile:
        for nutrient, info in nutrition_profile.items():
            # Use lowercase status with dashes for CSS class names
            status_class = f"status-{info['status'].lower().replace(' ', '-').replace('(', '').replace(')', '')}"
            html += f"""            <tr>
                <td>{nutrient}</td>
                <td>{info['amount']}</td>
                <td>{info['meal_percentage']}</td>
                <td>{info['daily_percentage']}</td>
                <td class="{status_class}">{info['status']}</td>
            </tr>
"""
    else:
         html += "            <tr><td colspan='5'>Nutrition profile data unavailable.</td></tr>\n"
    html += """
        </tbody>
    </table>
"""
    # Summary Box
    html += f"""
    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Nutrients at Good Levels:</strong> {summary.get('nutrients_at_good_levels', 0)} / {summary.get('total_nutrients_tracked', 0)}</p>
        <p><strong>Nutrients Below Target (<85%):</strong> {summary.get('nutrients_below_target', 0)} / {summary.get('total_nutrients_tracked', 0)}</p>
        <p><strong>Nutrients Above Target (>150%):</strong> {summary.get('nutrients_above_target', 0)} / {summary.get('total_nutrients_tracked', 0)}</p>
    </div>
"""
    # Penalties Box (Optional)
    if penalties:
        html += """
    <div class="penalties-box">
        <h2>Optimization Penalties Used</h2>
        <ul>
"""
        for key, value in penalties.items():
            html += f"            <li><strong>{key.replace('_', ' ').title()}:</strong> {value:.3f}x</li>\n"
        html += """
        </ul>
    </div>
"""
    # Close HTML
    html += """
</div>
</body>
</html>
"""
    return html


# --- CLI Specific Functions ---

def print_nutrition_report(foods_consumed, food_data_dict, rdi, nutrient_mapping, number_of_meals=1, meal_number=1):
    """Print a detailed nutrition report to the console (CLI Mode)."""
    meal_rdi = {nutrient: target / number_of_meals for nutrient, target in rdi.items() if target is not None and target > 0}
    daily_rdi = {nutrient: target for nutrient, target in rdi.items() if target is not None and target > 0}

    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}
    for food, amount in foods_consumed.items():
        if food not in food_data_dict: continue
        safe_amount = max(0, amount)
        food_details = food_data_dict[food]
        density = food_details.get('density', 100.0)
        if density <= 0: density = 100.0

        for nutrient in meal_rdi:
            nutrient_val = food_details.get(nutrient, 0.0)
            nutrient_per_gram = nutrient_val / density
            final_nutrients[nutrient] += nutrient_per_gram * safe_amount

    print(f"\n=== CONSOLE NUTRITION REPORT (Meal {meal_number}/{number_of_meals}) ===")
    print(f"(Meal targets ~{100/number_of_meals:.1f}% of daily needs)")

    print("\nRecommended Food Quantities:")
    if foods_consumed:
        for food, amount in sorted(foods_consumed.items(), key=lambda item: item[1], reverse=True):
            print(f"- {food}: {amount:.1f}g")
    else:
        print("- No foods in final solution.")

    print("\nNutrition Profile:")
    print(f"{'Nutrient':<35} {'Amount':<12} {'% Meal Target':<15} {'% Daily RDI':<15} {'Status'}")
    print("-" * 90)

    nutrients_low = 0
    nutrients_ok = 0
    nutrients_high = 0
    if final_nutrients and meal_rdi:
        for nutrient, amount in sorted(final_nutrients.items()):
            meal_target = meal_rdi.get(nutrient)
            daily_target = daily_rdi.get(nutrient)
            unit = _get_unit(nutrient, nutrient_mapping)

            meal_percentage_str = "N/A"
            daily_percentage_str = "N/A"
            status = "N/A"

            if meal_target is not None and meal_target > 0:
                meal_percentage = (amount / meal_target) * 100
                meal_percentage_str = f"{meal_percentage:.1f}%"
                if meal_percentage < 85:
                    status = "LOW"
                    nutrients_low += 1
                elif meal_percentage > 150:
                    status = "HIGH"
                    nutrients_high += 1
                else:
                    status = "OK"
                    nutrients_ok += 1
            else:
                 status = "No Target" # Nutrient tracked but no RDI target for it


            if daily_target is not None and daily_target > 0:
                daily_percentage = (amount / daily_target) * 100
                daily_percentage_str = f"{daily_percentage:.1f}%"


            print(f"{nutrient:<35} {amount:.1f}{unit:<9} {meal_percentage_str:<15} {daily_percentage_str:<15} {status}")
    else:
        print("Could not calculate nutrition profile.")

    print("\nSummary:")
    total_tracked = len(meal_rdi)
    print(f"- Good Levels (85-150%): {nutrients_ok}/{total_tracked}")
    print(f"- Below Target (<85%):   {nutrients_low}/{total_tracked}")
    print(f"- Above Target (>150%):  {nutrients_high}/{total_tracked}")
    print("-" * 90)


# --- Utility Functions (Index generation, Cleanup, Init Dirs) ---

def generate_index(recipes_base_dir="recipes"):
    """Generate both Markdown and HTML index files for meal plans"""
    logging.info("Generating index files...")
    meals = []
    json_dir = os.path.join(recipes_base_dir, "json")
    html_dir = os.path.join(recipes_base_dir, "html")

    if not os.path.isdir(json_dir):
        logging.warning(f"JSON directory not found: {json_dir}. Cannot generate index.")
        return

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                meal_info = data.get("meal_info", {})
                summary = data.get("summary", {})

                # Provide defaults for potentially missing keys
                meals.append({
                    "filename_html": os.path.splitext(filename)[0] + ".html",
                    "run_number": meal_info.get("run_number", "N/A"),
                    'timestamp': meal_info.get('date', "N/A"),
                    "diet_type": meal_info.get("diet_type", "unknown"),
                    "optimization_score": meal_info.get("optimization_score", float('inf')),
                    "nutrients_ok": summary.get("nutrients_at_good_levels", 0),
                    "nutrients_low": summary.get("nutrients_below_target", 0),
                    "nutrients_high": summary.get("nutrients_above_target", 0),
                    "food_items": meal_info.get("number_of_foods", 0),
                    "generations": meal_info.get("generations", 0),
                    "execution_time": meal_info.get("execution_time_seconds", 0),
                })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logging.warning(f"Skipping file {filename} due to error: {e}")
                continue

    # Sort meals by optimization score (ascending - better scores first)
    meals.sort(key=lambda x: x["optimization_score"])

    # --- Generate Markdown (README.md) ---
    md_path = "README.md"
    try:
        with open(md_path, "w", encoding='utf-8') as f:
            f.write("# Genetic Algorithm Optimised Nutrition Recipes\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Run # | Diet | Score | Foods | Nutrients (OK/Low/High) | Generations | Time (s) | HTML Report |\n")
            f.write("|-------|------|-------|-------|-------------------------|-------------|----------|-------------|\n")
            for meal in meals:
                nutrients_str = f"{meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']}"
                # Link relative to the recipes/html directory from root README
                html_link = os.path.join(recipes_base_dir, "html", meal['filename_html']).replace('\\', '/')
                f.write(f"| {meal['run_number']} | {meal['diet_type']} | {meal['optimization_score']:.2f} | "
                        f"{meal['food_items']} | {nutrients_str} | "
                        f"{meal['generations']} | {meal['execution_time']:.1f} | "
                        f"[{meal['filename_html']}]({html_link}) |\n")
        logging.info(f"Generated {md_path}")
    except IOError as e:
        logging.error(f"Failed to write {md_path}: {e}")


    # --- Generate HTML Index (recipes/html/index.html) ---
    html_index_path = os.path.join(html_dir, "index.html")
    try:
        with open(html_index_path, "w", encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Optimized Nutrition Recipes Index</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
        .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #0056b3; text-align: center; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9em; }}
        th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #e9ecef; font-weight: bold; cursor: pointer; }} /* Add cursor for sort */
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #f1f1f1; }}
        a {{ color: #0056b3; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .description {{ background-color: #e9f7ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #bce8f1; font-size: 0.95em;}}
        .score-good {{ color: #198754; }}
        .score-warn {{ color: #ffc107; }}
        .score-bad {{ color: #dc3545; font-weight: bold; }}
        /* Style for GitHub ForkMe SVG */
        .github-fork-ribbon {{ position: absolute; top: 0; right: 0; border: 0; z-index: 1000; }}
        @media (max-width: 768px) {{
            th, td {{ font-size: 0.85em; padding: 8px; }}
            h1 {{ font-size: 1.6em; }}
            .container {{ padding: 15px; }}
        }}
         @media (max-width: 480px) {{
             table, thead, tbody, th, td, tr {{ display: block; }}
             thead tr {{ position: absolute; top: -9999px; left: -9999px; }}
             tr {{ border: 1px solid #ccc; margin-bottom: 5px; }}
             td {{ border: none; border-bottom: 1px solid #eee; position: relative; padding-left: 50%; text-align: right; }}
             td:before {{ position: absolute; top: 6px; left: 6px; width: 45%; padding-right: 10px; white-space: nowrap; text-align: left; font-weight: bold; }}
             /* Define data labels */
             td:nth-of-type(1):before {{ content: "#"; }}
             td:nth-of-type(2):before {{ content: "Date"; }}
             td:nth-of-type(3):before {{ content: "Diet"; }}
             td:nth-of-type(4):before {{ content: "Score"; }}
             td:nth-of-type(5):before {{ content: "Foods"; }}
             td:nth-of-type(6):before {{ content: "Nutrients OK/L/H"; }}
             td:nth-of-type(7):before {{ content: "Gens"; }}
             td:nth-of-type(8):before {{ content: "Time(s)"; }}
             td:nth-of-type(9):before {{ content: "Recipe Link"; }}
             .github-fork-ribbon {{ display: none; }} /* Hide ribbon on small screens */
         }}
    </style>
</head>
<body>
<a href="https://github.com/alexlaverty/optimize-nutrition" target="_blank" class="github-fork-ribbon">
    <img loading="lazy" width="149" height="149" src="https://github.blog/wp-content/uploads/2008/12/forkme_right_darkblue_121621.png?resize=149%2C149" class="attachment-full size-full" alt="Fork me on GitHub" data-recalc-dims="1">
</a>
<div class="container">
    <h1>Optimized Nutrition Recipes Index</h1>
    <div class="description">
        <p>This table lists meal plans optimized using a genetic algorithm. Lower scores indicate a better match to nutritional targets based on the configured penalties. Click a column header to sort. Click the link in the 'Recipe Link' column to view the detailed HTML report for each meal.</p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <table id="recipesTable">
        <thead>
            <tr>
                <th onclick="sortTable(0)">#</th>
                <th onclick="sortTable(1)">Date</th>
                <th onclick="sortTable(2)">Diet</th>
                <th onclick="sortTable(3)">Score</th>
                <th onclick="sortTable(4)">Foods</th>
                <th onclick="sortTable(5)">Nutrients (OK/Low/High)</th>
                <th onclick="sortTable(6)">Gens</th>
                <th onclick="sortTable(7)">Time (s)</th>
                <th>Recipe Link</th>
            </tr>
        </thead>
        <tbody>
""")
            for idx, meal in enumerate(meals, start=1):
                score = meal['optimization_score']
                score_class = 'score-good' if score < 5 else 'score-warn' if score < 15 else 'score-bad'
                nutrients_str = f"{meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']}"
                # Link should be relative to the index.html file itself
                html_link = f"{meal['filename_html']}"
                f.write(f"""            <tr>
                <td>{idx}</td>
                <td>{meal['timestamp']}</td>
                <td>{meal['diet_type']}</td>
                <td class="{score_class}" data-sort="{score:.4f}">{score:.2f}</td>
                <td data-sort="{meal['food_items']}">{meal['food_items']}</td>
                <td>{nutrients_str}</td>
                <td data-sort="{meal['generations']}">{meal['generations']}</td>
                 <td data-sort="{meal['execution_time']:.2f}">{meal['execution_time']:.1f}</td>
                <td><a href="{html_link}">{meal['filename_html']}</a></td>
            </tr>
""")
            f.write("""        </tbody>
    </table>
</div>

<script>
function sortTable(n) {
  var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
  table = document.getElementById("recipesTable");
  switching = true;
  // Set the sorting direction to ascending:
  dir = "asc";
  /* Make a loop that will continue until no switching has been done: */
  while (switching) {
    // Start by saying: no switching is done:
    switching = false;
    rows = table.rows;
    /* Loop through all table rows (except the first, which contains table headers): */
    for (i = 1; i < (rows.length - 1); i++) {
      // Start by saying there should be no switching:
      shouldSwitch = false;
      /* Get the two elements you want to compare, one from current row and one from the next: */
      x = rows[i].getElementsByTagName("TD")[n];
      y = rows[i + 1].getElementsByTagName("TD")[n];
      /* Check if the two rows should switch place, based on the direction, asc or desc: */
      var xContent = x.dataset.sort || x.innerHTML.toLowerCase();
      var yContent = y.dataset.sort || y.innerHTML.toLowerCase();

      // Try converting to numbers for sorting if possible
      var xVal = parseFloat(xContent);
      var yVal = parseFloat(yContent);
      if (!isNaN(xVal) && !isNaN(yVal)) {
          xContent = xVal;
          yContent = yVal;
      }

      if (dir == "asc") {
        if (xContent > yContent) {
          // If so, mark as a switch and break the loop:
          shouldSwitch = true;
          break;
        }
      } else if (dir == "desc") {
        if (xContent < yContent) {
          // If so, mark as a switch and break the loop:
          shouldSwitch = true;
          break;
        }
      }
    }
    if (shouldSwitch) {
      /* If a switch has been marked, make the switch and mark that a switch has been done: */
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      // Each time a switch is done, increase this count by 1:
      switchcount ++;
    } else {
      /* If no switching has been done AND the direction is "asc", set the direction to "desc" and run the while loop again. */
      if (switchcount == 0 && dir == "asc") {
        dir = "desc";
        switching = true;
      }
    }
  }
}
// Initial sort by score (column 3) ascending
document.addEventListener('DOMContentLoaded', function() {
    sortTable(3); // Sort by score column initially
});
</script>

</body>
</html>
""")
        logging.info(f"Generated {html_index_path}")
    except IOError as e:
        logging.error(f"Failed to write {html_index_path}: {e}")


def init_directories(base_dir='recipes', subdirs=['json', 'html'], other_dirs=['rdi', 'diets']):
    """Initialize directory structure."""
    dirs_to_create = other_dirs[:]
    for subdir in subdirs:
        dirs_to_create.append(os.path.join(base_dir, subdir))

    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                logging.info(f"Created directory: {dir_path}")
            except OSError as e:
                logging.error(f"Failed to create directory {dir_path}: {e}")


def cleanup_high_score_recipes(max_score=20, max_files=250, recipes_base_dir="recipes"):
    """Remove recipe files based on score threshold and total file count limit."""
    logging.info(f"Cleaning up recipes (max score: {max_score}, max files: {max_files})...")
    recipes = []
    removed_count = 0
    json_dir = os.path.join(recipes_base_dir, "json")
    html_dir = os.path.join(recipes_base_dir, "html")

    if not os.path.isdir(json_dir):
        logging.warning(f"JSON directory not found: {json_dir}. Skipping cleanup.")
        return

    # Collect all recipe data
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                score = data.get("meal_info", {}).get("optimization_score", float('inf'))
                recipes.append({
                    "filename_base": os.path.splitext(filename)[0],
                    "score": score,
                    "json_filepath": filepath
                })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logging.warning(f"Error processing {filename} during cleanup: {e}")
                continue

    # Sort recipes by score (best scores first)
    recipes.sort(key=lambda x: x["score"])

    # Identify files to remove
    files_to_remove = []
    kept_count = 0
    for recipe in recipes:
        # Keep if score is good enough AND within file limit
        if recipe["score"] <= max_score and kept_count < max_files:
            kept_count += 1
        else:
            # Mark for removal otherwise
            files_to_remove.append(recipe)
            reason = f"score {recipe['score']:.2f} > {max_score}" if recipe["score"] > max_score else f"exceeds file limit ({kept_count+1} > {max_files})"
            #logging.debug(f"Marking {recipe['filename_base']} for removal ({reason})")


    # Remove marked files
    for recipe in files_to_remove:
        try:
            # Remove JSON
            os.remove(recipe["json_filepath"])
            # Remove corresponding HTML
            html_file = os.path.join(html_dir, f"{recipe['filename_base']}.html")
            if os.path.exists(html_file):
                os.remove(html_file)
            removed_count += 1
            # logging.info(f"Removed {recipe['filename_base']}") # Log if needed
        except OSError as e:
            logging.error(f"Error removing files for {recipe['filename_base']}: {e}")

    remaining_count = len(recipes) - removed_count
    logging.info(f"Cleanup complete: Removed {removed_count} recipes. {remaining_count} remaining.")
    if recipes:
        best_score = recipes[0]['score'] if remaining_count > 0 else recipes[0]['score'] # Show best overall even if removed
        worst_remaining_score = recipes[min(remaining_count - 1, len(recipes)-1)]['score'] if remaining_count > 0 else 'N/A'
        logging.info(f"Best score overall: {best_score:.2f}")
        if worst_remaining_score != 'N/A':
            logging.info(f"Worst score remaining: {worst_remaining_score:.2f}")


# --- Flask Routes and SocketIO Event Handlers ---

@app.route('/')
def index():
    """Serves the main HTML page for the Web UI."""
    logging.info("Serving index.html")
    # Render the template (Flask looks for it in a 'templates' folder)
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    logging.info(f"Client connected: {request.sid}")
    # Optionally send initial data or status
    emit('status_update', {'message': 'Connected to optimization server.'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    logging.info(f"Client disconnected: {request.sid}")

@socketio.on('start_optimization')
def handle_start_optimization(data):
    """Handles the start optimization request from the web UI."""
    sid = request.sid
    logging.info(f"Received start_optimization request from {sid}: {data}")
    emit('status_update', {'message': 'Received request. Preparing optimization...'}, room=sid)

    # --- Validate Input Data ---
    try:
        diet_type = data.get('diet_type', 'all')
        num_foods = int(data.get('num_foods', 20))
        generations = int(data.get('generations', 100))
        population_size = int(data.get('population_size', 50))
        number_of_meals = 1 # Hardcoded for now, could be an input
        meal_number = 1     # Hardcoded for now

        if not (10 <= num_foods <= 100): # Example validation
            raise ValueError("Number of foods must be between 10 and 100.")
        if not (10 <= generations <= 1000):
             raise ValueError("Generations must be between 10 and 1000.")
        if not (10 <= population_size <= 200):
             raise ValueError("Population size must be between 10 and 200.")

    except (TypeError, ValueError) as e:
        logging.error(f"Invalid input data from {sid}: {e}")
        emit('optimization_error', {'message': f"Invalid input: {e}"}, room=sid)
        return # Stop processing

    # --- Ensure Data is Loaded ---
    if loaded_data["df"] is None or loaded_data["nutrient_mapping"] is None:
        logging.error("Server data (Excel/RDI) not loaded. Cannot start optimization.")
        emit('optimization_error', {'message': 'Server error: Data files not loaded.'}, room=sid)
        return

    # --- Prepare Data for this Run ---
    df = loaded_data["df"]
    nutrient_mapping = loaded_data["nutrient_mapping"]
    rdi_values = loaded_data["rdi_values"]
    run_number = get_next_run_number()

    working_df = df.copy() # Work on a copy

    # Filter by diet type
    if diet_type != 'all':
        config_file = os.path.join('diets', f'{diet_type}.json')
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                diet_config = json.load(f)

            initial_food_count = len(working_df)
            emit('status_update', {'message': f'Applying "{diet_type}" diet filter...'}, room=sid)

            # Apply inclusion first
            if 'included_terms' in diet_config and diet_config['included_terms']:
                included_terms = [term.lower() for term in diet_config['included_terms']]
                inclusion_mask = working_df['Food Name'].str.lower().apply(
                    lambda x: any(term in str(x) for term in included_terms) if pd.notna(x) else False
                )
                working_df = working_df[inclusion_mask]
                logging.info(f"Run #{run_number}: Included {len(working_df)} foods after inclusion filter.")

            # Then apply exclusion
            if 'excluded_terms' in diet_config and diet_config['excluded_terms']:
                excluded_terms = [term.lower() for term in diet_config['excluded_terms']]
                exclusion_mask = ~working_df['Food Name'].str.lower().apply(
                    lambda x: any(term in str(x) for term in excluded_terms) if pd.notna(x) else False
                )
                working_df = working_df[exclusion_mask]
                logging.info(f"Run #{run_number}: {len(working_df)} foods remaining after exclusion filter.")

            emit('status_update', {'message': f'Filtered to {len(working_df)} foods for "{diet_type}".'}, room=sid)

            if len(working_df) == 0:
                logging.warning(f"Run #{run_number}: No foods remain after filtering for {diet_type} diet")
                emit('optimization_error', {'message': f'No foods found matching the "{diet_type}" filter.'}, room=sid)
                return
        except FileNotFoundError:
            logging.error(f"Diet config file not found: {config_file}")
            emit('optimization_error', {'message': f'Configuration for diet "{diet_type}" not found.'}, room=sid)
            return
        except Exception as e:
            logging.error(f"Error applying diet filter {diet_type}: {e}")
            emit('optimization_error', {'message': f'Error processing diet filter: {e}'}, room=sid)
            return

    # Select random subset of foods
    if len(working_df) < num_foods:
        logging.warning(f"Run #{run_number}: Requested {num_foods} but only {len(working_df)} available after filtering. Using all available.")
        emit('status_update', {'message': f'Warning: Only {len(working_df)} foods available after filtering.'}, room=sid)
        num_foods = len(working_df)
    elif num_foods <= 0:
         logging.error(f"Run #{run_number}: Invalid number of foods requested ({num_foods}).")
         emit('optimization_error', {'message': f'Cannot select {num_foods} foods.'}, room=sid)
         return

    random_foods_df = working_df.sample(n=num_foods)
    emit('status_update', {'message': f'Selected {num_foods} random foods. Starting optimization...'}, room=sid)

    # --- Start Optimization in Background Task ---
    # Pass necessary data and SocketIO instance/SID to the task
    task_args = (
        random_foods_df,
        nutrient_mapping,
        rdi_values,
        number_of_meals,
        meal_number,
        0.3, # randomness_factor - currently unused but kept placeholder
        population_size,
        generations,
        diet_type,
        run_number,
        socketio, # Pass the instance
        sid       # Pass the specific client's SID
    )
    # socketio.start_background_task is the correct way with Flask-SocketIO + eventlet/gevent
    #socketio.start_background_task(target=optimize_nutrition, *task_args)
    socketio.start_background_task(optimize_nutrition, *task_args)

    logging.info(f"Started background optimization task for SID: {sid}, Run #{run_number}")
    # Optionally, send confirmation back immediately
    # emit('optimization_started', {'run_number': run_number}, room=sid)


# --- Main Execution Block ---

def run_cli(args):
    """Runs the optimization in Command Line Interface mode."""
    logging.info("Running in CLI mode.")
    global_start_time = time.time()

    # --- Load Data ---
    df, nutrient_mapping, rdi_values = _load_data()
    if df is None or nutrient_mapping is None:
        logging.error("Failed to load necessary data. Exiting CLI mode.")
        return

    # --- Settings ---
    # Use args if provided, otherwise use random or fixed defaults
    num_meals_cli = 1
    meal_num_cli = 1
    pop_size_cli = 100 # Example fixed value for CLI

    # Diet types to run
    #diet_types_to_run = ['all', 'vegan', 'wfpb', 'nutrient_dense']
    diet_types_to_run = ['nutrient_dense'] # Quick test

    for diet_type in diet_types_to_run:
        run_start_time = time.time()
        logging.info(f"\n=== Starting CLI Optimization: {diet_type.upper()} ===")

        working_df = df.copy()

        # Filter foods based on diet type (similar logic to web handler)
        if diet_type != 'all':
            config_file = os.path.join('diets', f'{diet_type}.json')
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    diet_config = json.load(f)
                initial_count = len(working_df)
                logging.info(f"Applying '{diet_type}' filter (Initial: {initial_count} foods)...")

                if 'included_terms' in diet_config and diet_config['included_terms']:
                    included_terms = [term.lower() for term in diet_config['included_terms']]
                    inclusion_mask = working_df['Food Name'].str.lower().apply(
                        lambda x: any(term in str(x) for term in included_terms) if pd.notna(x) else False
                    )
                    working_df = working_df[inclusion_mask]

                if 'excluded_terms' in diet_config and diet_config['excluded_terms']:
                    excluded_terms = [term.lower() for term in diet_config['excluded_terms']]
                    exclusion_mask = ~working_df['Food Name'].str.lower().apply(
                        lambda x: any(term in str(x) for term in excluded_terms) if pd.notna(x) else False
                    )
                    working_df = working_df[exclusion_mask]

                logging.info(f"Filtered to {len(working_df)} foods for '{diet_type}'.")

                if len(working_df) == 0:
                    logging.warning(f"No foods remain after filtering for {diet_type}. Skipping.")
                    continue
            except FileNotFoundError:
                logging.warning(f"Diet config not found: {config_file}. Skipping {diet_type}.")
                continue
            except Exception as e:
                 logging.error(f"Error filtering for {diet_type}: {e}. Skipping.")
                 continue

        # Determine parameters for this run
        n_foods = args.foods if args.foods is not None else random.randint(15, 35)
        generations = args.generations if args.generations is not None else random.randint(100, 300)
        #generations = 3 # Force quick test

        if len(working_df) < n_foods:
            logging.warning(f"Requested {n_foods} foods, but only {len(working_df)} available for {diet_type}. Using all.")
            n_foods = len(working_df)
        if n_foods <= 0:
             logging.error(f"Cannot run optimization with {n_foods} foods. Skipping {diet_type}.")
             continue

        random_foods_df = working_df.sample(n=n_foods)
        logging.info(f"Selected {n_foods} random foods. Generations: {generations}. Population: {pop_size_cli}.")

        # Get run number before optimization call
        current_run_number = get_next_run_number()

        # Run optimization (without socketio instance)
        result = optimize_nutrition(
            food_df=random_foods_df,
            nutrient_mapping=nutrient_mapping,
            rdi_targets=rdi_values,
            number_of_meals=num_meals_cli,
            meal_number=meal_num_cli,
            population_size=pop_size_cli,
            generations=generations,
            diet_type=diet_type,
            run_number=current_run_number,
             # No socketio/sid for CLI
            socketio_instance=None,
            sid=None
        )

        if result and result.get("solution"):
            # Optionally print the console report for CLI runs
            # Note: The 'available_foods' structure used by print_nutrition_report is created
            # inside optimize_nutrition. We need to pass the final solution and the original
            # data structures it was based on.
            # Recreate `available_foods` based on the selected `random_foods_df`
            # to pass to print function (this is slightly redundant)
            cli_available_foods = {}
            nutrient_cols_in_df = [col for col in nutrient_mapping.keys() if col in random_foods_df.columns]
            for idx, row in random_foods_df.iterrows():
                 food_name = row['Food Name']
                 if isinstance(food_name, str):
                     food_data = {'density': 100}
                     for nutrient_col in nutrient_cols_in_df:
                         food_data[nutrient_col] = pd.to_numeric(row[nutrient_col], errors='coerce')
                         if pd.isna(food_data[nutrient_col]): food_data[nutrient_col] = 0.0
                     cli_available_foods[food_name] = food_data

            print_nutrition_report(result["solution"], cli_available_foods, rdi_values, nutrient_mapping, num_meals_cli, meal_num_cli)

            logging.info(f"=== Completed CLI Optimization: {diet_type.upper()} (Time: {time.time() - run_start_time:.2f}s) ===\n")
        else:
            logging.error(f"Optimization failed for {diet_type.upper()}.")
            logging.info(f"=== Aborted CLI Optimization: {diet_type.upper()} ===\n")


    # --- Post-run tasks for CLI ---
    logging.info("CLI run finished. Performing cleanup and index generation.")
    cleanup_high_score_recipes(max_score=25, max_files=300) # Example cleanup params
    generate_index()
    logging.info(f"Total CLI execution time: {time.time() - global_start_time:.2f} seconds")


def run_webui():
    """Loads data and starts the Flask-SocketIO web server."""
    logging.info("Running in Web UI mode.")

    # --- Load Data Globally for Flask App ---
    df, nutrient_mapping, rdi_values = _load_data()
    if df is None or nutrient_mapping is None:
        logging.error("CRITICAL: Failed to load data for Web UI. Server cannot start properly.")
        # Decide whether to exit or run with limited functionality
        # For now, we'll attempt to run but optimization will fail.
        # exit(1) # Or just log the error and let it run
    else:
        loaded_data["df"] = df
        loaded_data["nutrient_mapping"] = nutrient_mapping
        loaded_data["rdi_values"] = rdi_values
        logging.info("Data loaded successfully for Web UI.")

    # --- Start Server ---
    # Use eventlet (or gevent) as recommended by Flask-SocketIO for production/async
    logging.info("Starting Flask-SocketIO server...")
    # Use host='0.0.0.0' to make it accessible on the network
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) # Turn debug=False for production


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Optimize nutrition using genetic algorithm (CLI or Web UI)')
    parser.add_argument('--webui', action='store_true', help='Run as a Flask web UI instead of CLI')
    parser.add_argument('--generations', type=int, default=None, help='(CLI Mode) Number of generations (default: random)')
    parser.add_argument('--foods', type=int, default=None, help='(CLI Mode) Number of foods to select (default: random)')
    # Add host/port args for web UI if needed
    # parser.add_argument('--host', default='127.0.0.1', help='(Web UI Mode) Host for the web server')
    # parser.add_argument('--port', type=int, default=5000, help='(Web UI Mode) Port for the web server')
    args = parser.parse_args()

    # --- Initial Setup ---
    init_directories() # Ensure directories exist regardless of mode

    # --- Mode Selection ---
    if args.webui:
        # Import eventlet here if needed specifically for running the server
        # import eventlet
        # eventlet.monkey_patch() # Necessary for eventlet async mode with standard libraries
        run_webui()
    else:
        run_cli(args)