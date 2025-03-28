# --- Attempt Eventlet Monkey Patching FIRST ---
# This MUST happen before importing libraries like Flask, SocketIO, requests etc.
try:
    import eventlet
    eventlet.monkey_patch()
    print("INFO: Eventlet monkey patching applied successfully.") # Optional: Confirmation
except ImportError:
    print("WARNING: Eventlet not found. Web server might have limited concurrency. Install with: pip install eventlet")
except Exception as e:
    print(f"ERROR: An unexpected error occurred during eventlet monkey patching: {e}")
    # Decide if you want to exit or continue without patching
    # exit(1) # Uncomment to exit if patching fails critically

# --- Now import other modules ---
from datetime import datetime
import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time
import logging
import threading

# --- Flask and SocketIO Imports ---
from flask import Flask, render_template, request, jsonify, send_from_directory # Added send_from_directory
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
            # Use pd.to_numeric to handle potential non-numeric values safely
            max_run = pd.to_numeric(df['run_number'], errors='coerce').max()
            return int(max_run) + 1 if pd.notna(max_run) else 1
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
        # Ensure RDI values are numeric, default to 0 if missing or invalid
        rdi_values = {
            nutrient: pd.to_numeric(details.get('rdi'), errors='coerce')
            for nutrient, details in nutrient_mapping.items()
        }
        # Filter out nutrients where RDI is NaN or None after conversion
        rdi_values = {k: v for k, v in rdi_values.items() if pd.notna(v)}
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

    # --- Input Validation ---
    if meal_number < 1 or meal_number > number_of_meals:
        raise ValueError(f"Meal number must be between 1 and {number_of_meals}")

    if food_df is None or food_df.empty:
        logging.error(f"Run #{run_number}: Food DataFrame is empty or None. Cannot optimize.")
        if socketio_instance and sid:
             socketio_instance.emit('optimization_error', {'message': 'No food data available for optimization.'}, room=sid)
        return None # Indicate failure

    if not nutrient_mapping or not rdi_targets:
        logging.error(f"Run #{run_number}: Nutrient mapping or RDI targets are missing.")
        if socketio_instance and sid:
             socketio_instance.emit('optimization_error', {'message': 'Nutrient mapping or RDI targets missing.'}, room=sid)
        return None

    logging.info(f"Starting optimization run #{run_number} for diet '{diet_type}' with {generations} generations, {population_size} population.")

    # --- RDI Target Processing ---
    # Ensure RDI targets only include nutrients present in mapping and are valid numbers
    valid_rdi_targets = {
        k: v for k, v in rdi_targets.items()
        if k in nutrient_mapping and pd.notna(v) and v > 0 # Ensure RDI target is positive
    }
    if len(valid_rdi_targets) != len(rdi_targets):
        logging.warning(f"Run #{run_number}: Some RDI targets were invalid, removed, or not in the nutrient mapping.")
    if not valid_rdi_targets:
        logging.error(f"Run #{run_number}: No valid RDI targets found after filtering.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'No valid RDI targets found.'}, room=sid)
        return None
    meal_rdi_targets = {nutrient: target / number_of_meals for nutrient, target in valid_rdi_targets.items()}

    # --- Food Data Preparation ---
    available_foods = {}
    nutrient_cols_in_df = [col for col in meal_rdi_targets.keys() if col in food_df.columns] # Use meal_rdi_targets keys now
    if not nutrient_cols_in_df:
         logging.error(f"Run #{run_number}: None of the target nutrients are present as columns in the provided food_df.")
         if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'Mismatch between target nutrients and food data columns.'}, room=sid)
         return None

    processed_food_names = [] # Keep track of names successfully processed
    for idx, row in food_df.iterrows():
        food_name = row.get('Food Name') # Use .get() for safety
        if not isinstance(food_name, str) or not food_name.strip():
             logging.warning(f"Run #{run_number}: Skipping food with invalid/empty name at index {idx}: '{food_name}'")
             continue

        food_name = food_name.strip() # Clean name
        food_data = {'density': 100} # Assume 100g base unit
        valid_food = True
        for nutrient_col in nutrient_cols_in_df:
            value = pd.to_numeric(row.get(nutrient_col), errors='coerce')
            food_data[nutrient_col] = 0.0 if pd.isna(value) else float(value) # Ensure standard float

        available_foods[food_name] = food_data
        processed_food_names.append(food_name)

    if not available_foods:
        logging.error(f"Run #{run_number}: No valid foods could be processed from the input DataFrame.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'No valid foods found in the selected data.'}, room=sid)
        return None

    # *** NEW: Emit the list of foods being optimized ***
    if socketio_instance and sid:
        try:
            # Sort food names alphabetically for consistent display
            sorted_food_names = sorted(list(available_foods.keys()))
            socketio_instance.emit('initial_foods', {'foods': sorted_food_names}, room=sid)
            logging.info(f"Run #{run_number}: Emitted initial food list ({len(sorted_food_names)} items) to {sid}")
        except Exception as e:
            logging.error(f"Run #{run_number}: Error emitting initial food list: {e}")


    # --- Genetic Algorithm Setup ---
    def create_solution():
        """Create a random solution (diet plan)."""
        return {food: max(0.0, random.uniform(5, 120)) for food in available_foods} # Use floats, smaller min

    # Generate random penalties for this optimization run
    penalties = {
        "under_rdi": random.uniform(1.8, 3.5), # Slightly higher penalty for under
        "over_rdi": random.uniform(0.1, 0.4),
        "over_ul": random.uniform(1.5, 3.0),
        "water_soluble": random.uniform(0.05, 0.15),
        "fat_soluble": random.uniform(0.2, 0.5)
    }
    logging.info(f"Run #{run_number} Penalties: {penalties}")
    if socketio_instance and sid:
        socketio_instance.emit('status_update', {'message': f"Generated penalties (under/over RDI/UL/water/fat): {penalties['under_rdi']:.2f}x / {penalties['over_rdi']:.2f}x / {penalties['over_ul']:.2f}x / {penalties['water_soluble']:.2f}x / {penalties['fat_soluble']:.2f}x"}, room=sid)

    def evaluate_solution(solution):
        """Evaluate how close the solution is to the RDI targets."""
        current_nutrients = {nutrient: 0.0 for nutrient in meal_rdi_targets} # Use floats
        for food, amount in solution.items():
            if food not in available_foods: continue
            safe_amount = max(0.0, amount) # Ensure non-negative float
            food_details = available_foods[food]
            density = food_details.get('density', 100.0)
            if density <= 0: density = 100.0 # Avoid division by zero

            for nutrient in meal_rdi_targets:
                nutrient_val = food_details.get(nutrient, 0.0)
                nutrient_per_gram = nutrient_val / density
                current_nutrients[nutrient] += nutrient_per_gram * safe_amount
        # Pass nutrient_mapping for _get_nutrient_type
        return _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties, nutrient_mapping)

    def mutate_solution(solution):
        """Mutate the solution by tweaking food amounts."""
        if not solution: return solution
        food_to_mutate = random.choice(list(solution.keys()))
        change_factor = random.uniform(-0.25, 0.25) # Percentage change
        absolute_change = random.uniform(-15, 15)   # Absolute change
        solution[food_to_mutate] = max(0.0, solution[food_to_mutate] * (1 + change_factor) + absolute_change)
        return solution

    def crossover_solution(sol1, sol2):
        """Create a new solution by combining two parent solutions (e.g., blend crossover)."""
        new_solution = {}
        all_foods = list(set(sol1.keys()) | set(sol2.keys()))
        for food in all_foods:
            alpha = random.random() # Blend factor
            amount1 = sol1.get(food, 0.0)
            amount2 = sol2.get(food, 0.0)
            new_solution[food] = max(0.0, alpha * amount1 + (1 - alpha) * amount2)
        return new_solution

    # --- Genetic Algorithm Execution ---
    population = [create_solution() for _ in range(population_size)]
    best_overall_solution = population[0] # Initialize with first one
    best_overall_score = evaluate_solution(best_overall_solution)
    history_data = [] # Collect history for CSV saving

    for generation in range(generations):
        # Evaluate population
        scores = []
        for sol in population:
            try:
                score = evaluate_solution(sol)
                # Ensure score is a valid number, handle potential errors during evaluation
                if pd.notna(score):
                    scores.append((float(score), sol)) # Store score as float
                else:
                    logging.warning(f"Run #{run_number} Gen {generation+1}: Invalid score calculated for a solution. Skipping.")
            except Exception as e:
                logging.error(f"Run #{run_number} Gen {generation+1}: Error evaluating solution: {e}. Skipping solution.")
                continue # Skip this problematic solution

        if not scores:
            logging.error(f"Run #{run_number} Generation {generation+1}: No valid scores generated in this generation. Stopping.")
            if socketio_instance and sid:
                 socketio_instance.emit('optimization_error', {'message': 'Evaluation failed to produce scores.'}, room=sid)
            break # Stop if evaluation fails

        scores.sort(key=lambda x: x[0])
        current_best_score, current_best_solution = scores[0]

        # Update overall best if current generation is better
        if current_best_score < best_overall_score:
            best_overall_score = current_best_score
            best_overall_solution = current_best_solution # Keep the actual best solution found so far

        # --- Real-time Update & History Logging ---
        timestamp_now = datetime.now()
        gen_info = {
            'run_number': int(run_number), # Ensure int
            'generation': generation + 1,
            'score': float(current_best_score), # Ensure float
            'timestamp': timestamp_now.strftime('%Y-%m-%d %H:%M:%S')
        }
        history_data.append(gen_info)

        # Emit progress to the specific web client, if applicable
        if socketio_instance and sid:
            # *** MODIFIED: Include food amounts in the update ***
            # Ensure amounts are standard floats and rounded for display
            food_amounts_update = {
                food: round(float(amount), 1) # Round to 1 decimal place
                for food, amount in current_best_solution.items()
            }

            socketio_instance.emit('generation_update', {
                'generation': generation + 1,
                'score': float(current_best_score), # Send as float
                'total_generations': generations,
                'food_amounts': food_amounts_update # <-- Send current best amounts
            }, room=sid)
            # Give the server a tiny break to handle IO
            socketio.sleep(0.01)

        # Log progress to console
        if (generation + 1) % 10 == 0 or generation == generations - 1:
             logging.info(f"Run #{run_number} - Gen {generation + 1}/{generations}, Best Score: {current_best_score:.4f}")


        # --- Selection, Crossover, Mutation ---
        num_elites = max(1, int(population_size * 0.1)) # Keep top 10%
        elites = [sol for _, sol in scores[:num_elites]]

        # Tournament Selection for parents
        selected_parents = []
        tournament_size = 3
        for _ in range(population_size - num_elites): # Need parents for the rest
             tournament = random.sample(scores, k=min(tournament_size, len(scores)))
             winner = min(tournament, key=lambda x: x[0])[1] # Select the best solution (index 1) from the tournament
             selected_parents.append(winner)

        # Create new population
        new_population = elites[:] # Start with the elites
        while len(new_population) < population_size:
            if len(selected_parents) >= 2:
                parent1, parent2 = random.sample(selected_parents, 2)
            elif len(selected_parents) == 1: # Handle edge case
                 parent1 = parent2 = selected_parents[0]
            else: # Fallback if selection failed (shouldn't happen with valid scores)
                 parent1 = parent2 = create_solution()

            child = crossover_solution(parent1, parent2)
            if random.random() < 0.15: # Mutation probability (e.g., 15%)
                child = mutate_solution(child)
            new_population.append(child)

        population = new_population

    # --- Save History ---
    if history_data:
        history_df = pd.DataFrame(history_data)
        history_file = 'optimization_history.csv'
        try:
            # Ensure file exists with header or append correctly
            file_exists = os.path.exists(history_file)
            history_df.to_csv(history_file, mode='a', header=not file_exists, index=False)
        except IOError as e:
            logging.error(f"Run #{run_number}: Error saving optimization history to CSV: {e}")
        except Exception as e:
             logging.error(f"Run #{run_number}: Unexpected error saving history: {e}")

    # --- Post Optimization ---
    execution_time = time.time() - start_time
    logging.info(f"Optimization run #{run_number} finished. Execution time: {execution_time:.2f} seconds. Final Best Score: {best_overall_score:.4f}")

    if best_overall_solution is None:
        logging.error(f"Run #{run_number}: No best solution found after optimization loop.")
        if socketio_instance and sid:
            socketio_instance.emit('optimization_error', {'message': 'Optimization finished but no solution was found.'}, room=sid)
        return None

    # --- Generate Report with the actual best solution found ---
    try:
        report_paths = save_nutrition_report(
            best_overall_solution, # Use the best found across all generations
            available_foods,
            valid_rdi_targets,
            nutrient_mapping,
            best_overall_score, # Report the best score achieved
            run_number,
            number_of_meals,
            meal_number,
            generations,
            execution_time,
            diet_type,
            penalties
        )
        logging.info(f"Run #{run_number}: Report saved: JSON={report_paths.get('json', 'Failed')}, HTML={report_paths.get('html', 'Failed')}")
    except Exception as e:
        logging.error(f"Run #{run_number}: Error generating or saving report: {e}")
        report_paths = {'json': None, 'html': None} # Indicate failure

    # --- Emit Completion to Web UI ---
    if socketio_instance and sid:
        html_report_name = os.path.basename(report_paths['html']) if report_paths.get('html') else None
        json_report_name = os.path.basename(report_paths['json']) if report_paths.get('json') else None

        emit_data = {
            'message': f"Optimization complete! Score: {best_overall_score:.2f}",
            'report_html': html_report_name,
            'report_json': json_report_name,
            'run_number': int(run_number)
        }
        socketio_instance.emit('optimization_complete', emit_data, room=sid)

    # Round the amounts for the final returned dictionary
    final_solution = {
        food: round(float(amount)) # Ensure float then round to nearest int
        for food, amount in best_overall_solution.items()
        if round(float(amount)) > 1 # Filter very small amounts (e.g., < 1g after rounding)
    }

    return {
        "solution": final_solution,
        "score": float(best_overall_score), # Return as float
        "penalties": penalties,
        "report_paths": report_paths,
        "run_number": int(run_number) # Return as int
    }


# ... (rest of the functions _is_nutrition_sufficient, _get_nutrient_type, _calculate_nutrition_score, _get_unit remain largely the same) ...

def _is_nutrition_sufficient(current, targets, threshold=0.90):
    """Check if current nutrients are at least threshold percent of targets."""
    for nutrient, target in targets.items():
        if target > 0 and current.get(nutrient, 0) < threshold * target:
            return False
    return True

def _get_nutrient_type(nutrient_name, nutrient_mapping):
    """Determine if a nutrient is water-soluble, fat-soluble, or other based on mapping."""
    details = nutrient_mapping.get(str(nutrient_name), {}) # Ensure nutrient_name is string
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
    score = 0.0 # Use float
    ul_targets = {
        nutrient: pd.to_numeric(details.get('ul'), errors='coerce')
        for nutrient, details in nutrient_mapping.items()
    } # Load UL values, ensure numeric

    for nutrient, target in meal_rdi_targets.items():
        if target is None or target <= 0: continue # Skip nutrients with no target or invalid target

        current_value = current_nutrients.get(nutrient, 0.0)
        nutrient_type = _get_nutrient_type(nutrient, nutrient_mapping)
        upper_limit = ul_targets.get(nutrient)

        # --- Penalty for being UNDER target ---
        if current_value < target:
            deficit_ratio = (target - current_value) / target
            score += (deficit_ratio ** 2) * penalties["under_rdi"]

        # --- Penalty for being OVER target ---
        else:
            overage_ratio = (current_value - target) / target
            over_ul_penalty = 0.0

            # Check against Upper Limit (UL) if available and valid
            if pd.notna(upper_limit) and upper_limit > 0 and current_value > upper_limit:
                 ul_overage_ratio = (current_value - upper_limit) / upper_limit
                 over_ul_penalty = (ul_overage_ratio ** 2) * penalties["over_ul"]

            # Determine base penalty based on nutrient type
            if nutrient_type == 'water_soluble':
                base_overage_penalty = penalties["water_soluble"]
            elif nutrient_type == 'fat_soluble':
                base_overage_penalty = penalties["fat_soluble"]
            else: # Macronutrients, minerals etc.
                base_overage_penalty = penalties["over_rdi"]

            general_overage_penalty = (overage_ratio ** 2) * base_overage_penalty
            score += max(general_overage_penalty, over_ul_penalty)

    return score


def _get_unit(nutrient_name, nutrient_mapping):
    """Get the unit for a nutrient from the mapping."""
    return nutrient_mapping.get(str(nutrient_name), {}).get('unit', '') # Ensure key is string


def save_nutrition_report(foods_consumed, food_data_dict, rdi, nutrient_mapping, score, run_number,
                         number_of_meals=1, meal_number=1, generations=100,
                         execution_time=0, diet_type='all', penalties=None):
    """Save a detailed nutrition report as JSON and HTML."""
    report_timestamp = datetime.now()
    timestamp_str = report_timestamp.strftime('%Y%m%d_%H%M%S')
    report_date_str = report_timestamp.strftime('%Y-%m-%d')

    init_directories() # Ensure directories exist

    # Ensure targets are floats for calculations
    daily_rdi = {str(k): float(v) for k, v in rdi.items() if pd.notna(v) and v > 0}
    meal_rdi = {k: v / number_of_meals for k, v in daily_rdi.items()}

    # Calculate final nutrients
    final_nutrients = {nutrient: 0.0 for nutrient in meal_rdi} # Use floats
    foods_in_report = {} # Store amounts actually used for the report

    for food, amount in foods_consumed.items():
        food_str = str(food) # Ensure string key
        amount_float = float(amount) # Ensure float value
        if food_str not in food_data_dict or amount_float <= 0.01: continue # Skip if not in data or negligible

        foods_in_report[food_str] = amount_float # Add to report list

        safe_amount = max(0.0, amount_float)
        food_details = food_data_dict[food_str]
        density = food_details.get('density', 100.0)
        if density <= 0: density = 100.0

        for nutrient in meal_rdi: # Iterate through target nutrients
            nutrient_str = str(nutrient) # Ensure string key
            nutrient_val = food_details.get(nutrient_str, 0.0) # Get value, default 0
            nutrient_per_gram = nutrient_val / density
            final_nutrients[nutrient_str] += nutrient_per_gram * safe_amount

    # --- Build Report Dictionary ---
    report = {
        "meal_info": {
            "run_number": int(run_number),
            "diet_type": str(diet_type),
            "target_percentage": float(100/number_of_meals),
            "optimization_score": float(score),
            "generations": int(generations),
            "number_of_foods": len(foods_in_report), # Count foods actually used
            "execution_time_seconds": float(execution_time),
            "date": report_date_str,
            "timestamp": timestamp_str,
            "penalties": {str(k): float(f"{v:.3f}") for k, v in penalties.items()} if penalties else {}
        },
        # Use foods_in_report for quantities, sort by amount desc
        "food_quantities": {
            food: f"{amount:.1f}g"
            for food, amount in sorted(foods_in_report.items(), key=lambda item: item[1], reverse=True)
        },
        "nutrition_profile": {},
        "summary": {}
    }

    # Populate Nutrition Profile
    nutrients_low = 0
    nutrients_ok = 0
    nutrients_high = 0
    total_tracked = 0
    for nutrient, amount in final_nutrients.items():
        if nutrient not in meal_rdi: continue # Skip if somehow not a target nutrient

        total_tracked += 1
        meal_target = meal_rdi.get(nutrient)
        daily_target = daily_rdi.get(nutrient)
        unit = _get_unit(nutrient, nutrient_mapping)

        # Ensure targets are valid before calculating percentages
        meal_percentage = 0.0
        if meal_target is not None and meal_target > 0:
             meal_percentage = (amount / meal_target) * 100

        daily_percentage = 0.0
        if daily_target is not None and daily_target > 0:
             daily_percentage = (amount / daily_target) * 100

        status = "OK"
        if meal_percentage < 85:
             status = "LOW"
             nutrients_low += 1
        elif meal_percentage > 150:
             status = "HIGH"
             nutrients_high += 1
        else:
            nutrients_ok += 1

        report["nutrition_profile"][str(nutrient)] = {
            "amount": f"{amount:.1f}{unit}",
            "meal_percentage": f"{meal_percentage:.1f}%",
            "daily_percentage": f"{daily_percentage:.1f}%",
            "status": status
        }

    report["summary"] = {
        "nutrients_at_good_levels": nutrients_ok,
        "nutrients_below_target": nutrients_low,
        "nutrients_above_target": nutrients_high,
        "total_nutrients_tracked": total_tracked
    }


    # --- Save JSON ---
    json_filename = os.path.join('recipes', 'json', f'meal_{run_number}_{timestamp_str}.json')
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Run #{run_number}: Failed to save JSON report {json_filename}: {e}")
        json_filename = None # Indicate failure
    except TypeError as e:
        logging.error(f"Run #{run_number}: JSON serialization error for {json_filename}: {e}")
        json_filename = None

    # --- Generate and Save HTML ---
    html_filename = os.path.join('recipes', 'html', f'meal_{run_number}_{timestamp_str}.html')
    try:
        html_content = generate_html_report(report) # Use helper function
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except IOError as e:
        logging.error(f"Run #{run_number}: Failed to save HTML report {html_filename}: {e}")
        html_filename = None # Indicate failure
    except Exception as e:
         logging.error(f"Run #{run_number}: Failed to generate HTML report {html_filename}: {e}")
         html_filename = None

    return {"json": json_filename, "html": html_filename}


# ... (generate_html_report remains the same) ...
def generate_html_report(report_data):
    """Generates HTML content from the report dictionary."""
    meal_info = report_data.get('meal_info', {})
    food_quantities = report_data.get('food_quantities', {})
    nutrition_profile = report_data.get('nutrition_profile', {})
    summary = report_data.get('summary', {})
    penalties = meal_info.get('penalties', {})

    # Safely get values with defaults
    run_number = meal_info.get('run_number', 'N/A')
    diet_type = meal_info.get('diet_type', 'N/A')
    report_date = meal_info.get('date', 'N/A')
    target_percentage = meal_info.get('target_percentage', 0)
    optimization_score = meal_info.get('optimization_score', float('inf'))
    generations = meal_info.get('generations', 'N/A')
    num_foods = meal_info.get('number_of_foods', 'N/A')
    exec_time = meal_info.get('execution_time_seconds', 0)

    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meal {run_number} - Nutrition Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.5; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: #f9f9f9; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-bottom: 15px;}}
        h1 {{ text-align: center; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.95em; }}
        th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #ddd; word-wrap: break-word; }} /* Added word-wrap */
        th {{ background-color: #e9ecef; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .status-low {{ color: #dc3545; font-weight: bold; }}
        .status-ok {{ color: #28a745; }}
        .status-high {{ color: #ffc107; font-weight: bold; }}
        .status-very-high-check-ul {{ color: #e83e8c; font-weight: bold; background-color: #f9d6e4; }}
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
            table {{ font-size: 0.9em; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Meal {run_number} Nutrition Report</h1>

    <div class="info-box">
        <h2>Meal Information</h2>
        <p><strong>Run Number:</strong> {run_number}</p>
        <p><strong>Diet Type:</strong> {diet_type}</p>
        <p><strong>Date Generated:</strong> {report_date}</p>
        <p><strong>Target Nutrition:</strong> {target_percentage:.0f}% of Daily RDI</p>
        <p><strong>Optimization Score:</strong> {optimization_score:.3f} (Lower is better)</p>
        <p><strong>Generations:</strong> {generations}</p>
        <p><strong>Foods Used:</strong> {num_foods}</p>
        <p><strong>Execution Time:</strong> {exec_time:.2f} seconds</p>
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
         html += "            <tr><td colspan='2'>No significant food quantities listed in final solution.</td></tr>\n"
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
                <th>% Meal Target</th>
                <th>% Daily RDI</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
    if nutrition_profile:
        # Sort nutrients alphabetically for consistent order
        for nutrient, info in sorted(nutrition_profile.items()):
            status_class = f"status-{info.get('status', 'unknown').lower().replace(' ', '-').replace('(', '').replace(')', '')}"
            html += f"""            <tr>
                <td>{nutrient}</td>
                <td>{info.get('amount', 'N/A')}</td>
                <td>{info.get('meal_percentage', 'N/A')}</td>
                <td>{info.get('daily_percentage', 'N/A')}</td>
                <td class="{status_class}">{info.get('status', 'N/A')}</td>
            </tr>
"""
    else:
         html += "            <tr><td colspan='5'>Nutrition profile data unavailable.</td></tr>\n"
    html += """
        </tbody>
    </table>
"""
    # Summary Box
    total_tracked = summary.get('total_nutrients_tracked', 0)
    html += f"""
    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Nutrients at Good Levels (85-150%):</strong> {summary.get('nutrients_at_good_levels', 0)} / {total_tracked}</p>
        <p><strong>Nutrients Below Target (<85%):</strong> {summary.get('nutrients_below_target', 0)} / {total_tracked}</p>
        <p><strong>Nutrients Above Target (>150%):</strong> {summary.get('nutrients_above_target', 0)} / {total_tracked}</p>
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


# ... (print_nutrition_report remains the same) ...
def print_nutrition_report(foods_consumed, food_data_dict, rdi, nutrient_mapping, number_of_meals=1, meal_number=1):
    """Print a detailed nutrition report to the console (CLI Mode)."""
    # Ensure targets are floats for calculations
    daily_rdi = {str(k): float(v) for k, v in rdi.items() if pd.notna(v) and v > 0}
    meal_rdi = {k: v / number_of_meals for k, v in daily_rdi.items()}

    final_nutrients = {nutrient: 0.0 for nutrient in meal_rdi} # Use floats
    foods_in_report = {} # Store amounts actually used

    for food, amount in foods_consumed.items():
        food_str = str(food) # Ensure string key
        amount_float = float(amount) # Ensure float value
        if food_str not in food_data_dict or amount_float <= 0.01: continue # Skip if not in data or negligible

        foods_in_report[food_str] = amount_float # Add to report list

        safe_amount = max(0.0, amount_float)
        food_details = food_data_dict[food_str]
        density = food_details.get('density', 100.0)
        if density <= 0: density = 100.0

        for nutrient in meal_rdi: # Iterate through target nutrients
            nutrient_str = str(nutrient) # Ensure string key
            nutrient_val = food_details.get(nutrient_str, 0.0) # Get value, default 0
            nutrient_per_gram = nutrient_val / density
            final_nutrients[nutrient_str] += nutrient_per_gram * safe_amount


    print(f"\n=== CONSOLE NUTRITION REPORT (Meal {meal_number}/{number_of_meals}) ===")
    print(f"(Meal targets ~{100/number_of_meals:.1f}% of daily needs)")

    print("\nRecommended Food Quantities:")
    if foods_in_report:
         # Sort by amount descending for printing
        for food, amount in sorted(foods_in_report.items(), key=lambda item: item[1], reverse=True):
            print(f"- {food}: {amount:.1f}g")
    else:
        print("- No significant food quantities in final solution.")

    print("\nNutrition Profile:")
    print(f"{'Nutrient':<35} {'Amount':<12} {'% Meal Target':<15} {'% Daily RDI':<15} {'Status'}")
    print("-" * 90)

    nutrients_low = 0
    nutrients_ok = 0
    nutrients_high = 0
    total_tracked = 0

    if final_nutrients and meal_rdi:
         # Sort nutrients alphabetically for consistent order
        for nutrient, amount in sorted(final_nutrients.items()):
            if nutrient not in meal_rdi: continue # Skip if somehow not a target nutrient

            total_tracked += 1
            meal_target = meal_rdi.get(nutrient)
            daily_target = daily_rdi.get(nutrient)
            unit = _get_unit(nutrient, nutrient_mapping)

            meal_percentage_str = "N/A"
            daily_percentage_str = "N/A"
            status = "N/A"

            # Ensure targets are valid before calculating percentages
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
                 status = "No Target"

            if daily_target is not None and daily_target > 0:
                daily_percentage = (amount / daily_target) * 100
                daily_percentage_str = f"{daily_percentage:.1f}%"


            print(f"{nutrient:<35} {amount:.1f}{unit:<9} {meal_percentage_str:<15} {daily_percentage_str:<15} {status}")
    else:
        print("Could not calculate nutrition profile.")

    print("\nSummary:")
    print(f"- Good Levels (85-150%): {nutrients_ok}/{total_tracked}")
    print(f"- Below Target (<85%):   {nutrients_low}/{total_tracked}")
    print(f"- Above Target (>150%):  {nutrients_high}/{total_tracked}")
    print("-" * 90)


# ... (generate_index, init_directories, cleanup_high_score_recipes remain the same) ...
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
                    "total_nutrients": summary.get("total_nutrients_tracked", 0), # Get total tracked
                    "food_items": meal_info.get("number_of_foods", 0),
                    "generations": meal_info.get("generations", 0),
                    "execution_time": meal_info.get("execution_time_seconds", 0),
                })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logging.warning(f"Skipping index generation for file {filename} due to error: {e}")
                continue

    # Sort meals by optimization score (ascending - better scores first)
    meals.sort(key=lambda x: x["optimization_score"])

    # --- Generate Markdown (README.md) ---
    md_path = "README.md"
    try:
        with open(md_path, "w", encoding='utf-8') as f:
            f.write("# Genetic Algorithm Optimised Nutrition Recipes\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Run # | Diet | Score | Foods | Nutrients (OK/Low/High/Total) | Generations | Time (s) | HTML Report |\n")
            f.write("|-------|------|-------|-------|-------------------------------|-------------|----------|-------------|\n")
            for meal in meals:
                 # Include total nutrients in the string
                nutrients_str = f"{meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']}/{meal['total_nutrients']}"
                html_link = os.path.join(recipes_base_dir, "html", meal['filename_html']).replace('\\', '/') # Use forward slashes for web links
                f.write(f"| {meal['run_number']} | {meal['diet_type']} | {meal['optimization_score']:.2f} | "
                        f"{meal['food_items']} | {nutrients_str} | "
                        f"{meal['generations']} | {meal['execution_time']:.1f} | "
                        f"[{meal['filename_html']}]({html_link}) |\n")
        logging.info(f"Generated {md_path}")
    except IOError as e:
        logging.error(f"Failed to write {md_path}: {e}")


    # --- Generate HTML Index (recipes/html/index.html) ---
    html_index_path = os.path.join(html_dir, "index.html")
    if not os.path.exists(html_dir):
         try:
             os.makedirs(html_dir)
             logging.info(f"Created directory: {html_dir}")
         except OSError as e:
              logging.error(f"Failed to create HTML directory {html_dir}: {e}. Cannot save index.html.")
              return # Stop if we can't create the directory

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
        .github-fork-ribbon {{ position: fixed; /* Changed to fixed */ top: 0; right: 0; border: 0; z-index: 1000; }}
        @media (max-width: 768px) {{
            th, td {{ font-size: 0.85em; padding: 8px; }}
            h1 {{ font-size: 1.6em; }}
            .container {{ padding: 15px; max-width: 95%; }} /* Adjust width */
        }}
         @media (max-width: 600px) {{ /* Adjusted breakpoint */
             table, thead, tbody, th, td, tr {{ display: block; }}
             thead tr {{ position: absolute; top: -9999px; left: -9999px; }}
             tr {{ border: 1px solid #ccc; margin-bottom: 5px; }}
             td {{ border: none; border-bottom: 1px solid #eee; position: relative; padding-left: 45%; /* Adjusted padding */ text-align: right; min-height: 30px; /* Ensure cells have height */ }}
             td:before {{ position: absolute; top: 6px; left: 6px; width: 40%; /* Adjusted width */ padding-right: 10px; white-space: nowrap; text-align: left; font-weight: bold; font-size: 0.9em; /* Slightly smaller label */ }}
             /* Define data labels */
             td:nth-of-type(1):before {{ content: "#"; }}
             td:nth-of-type(2):before {{ content: "Date"; }}
             td:nth-of-type(3):before {{ content: "Diet"; }}
             td:nth-of-type(4):before {{ content: "Score"; }}
             td:nth-of-type(5):before {{ content: "Foods"; }}
             td:nth-of-type(6):before {{ content: "Nutrients"; }} /* Shortened Label */
             td:nth-of-type(7):before {{ content: "Gens"; }}
             td:nth-of-type(8):before {{ content: "Time(s)"; }}
             td:nth-of-type(9):before {{ content: "Recipe"; }} /* Shortened Label */
             .github-fork-ribbon {{ display: none; }} /* Hide ribbon on small screens */
         }}
    </style>
</head>
<body>
<!-- Updated GitHub Ribbon Link/Image -->
<a href="https://github.com/alexlaverty/optimize-nutrition" target="_blank" class="github-fork-ribbon" aria-label="Fork me on GitHub">
    <img loading="lazy" width="149" height="149" src="https://github.blog/wp-content/uploads/2008/12/forkme_right_darkblue_121621.png?resize=149%2C149" alt="Fork me on GitHub">
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
                <th onclick="sortTable(5)">Nutrients (OK/L/H/Tot)</th> <!-- Updated Header -->
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
                 # Include total nutrients in the string
                nutrients_str = f"{meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']}/{meal['total_nutrients']}"
                # Link should be relative to the index.html file itself
                html_link = f"{meal['filename_html']}"
                f.write(f"""            <tr>
                <td>{idx}</td>
                <td>{meal['timestamp']}</td>
                <td>{meal['diet_type']}</td>
                <td class="{score_class}" data-sort="{score:.4f}">{score:.2f}</td>
                <td data-sort="{meal['food_items']}">{meal['food_items']}</td>
                <td>{nutrients_str}</td> <!-- Display updated nutrient string -->
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
  dir = table.rows[0].getElementsByTagName("TH")[n].getAttribute("data-sort-dir") || "asc"; // Get current direction or default to asc

  // Toggle direction for next click
  table.rows[0].getElementsByTagName("TH")[n].setAttribute("data-sort-dir", dir === "asc" ? "desc" : "asc");

  while (switching) {
    switching = false;
    rows = table.rows;
    for (i = 1; i < (rows.length - 1); i++) {
      shouldSwitch = false;
      x = rows[i].getElementsByTagName("TD")[n];
      y = rows[i + 1].getElementsByTagName("TD")[n];
      var xContent = x.dataset.sort || x.textContent || x.innerText || ""; // Use textContent/innerText as fallback
      var yContent = y.dataset.sort || y.textContent || y.innerText || ""; // Use textContent/innerText as fallback

      // Handle numeric comparison robustly
      var xVal = parseFloat(xContent);
      var yVal = parseFloat(yContent);
      var compareResult;
      if (!isNaN(xVal) && !isNaN(yVal)) {
          compareResult = xVal - yVal; // Numeric comparison
      } else {
          // String comparison (case-insensitive)
          compareResult = xContent.toLowerCase().localeCompare(yContent.toLowerCase());
      }

      if ((dir === "asc" && compareResult > 0) || (dir === "desc" && compareResult < 0)) {
          shouldSwitch = true;
          break;
      }
    }
    if (shouldSwitch) {
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      switchcount++;
    } else {
      // If no switch and ascending, allow loop to exit (or switch to desc if needed, handled by toggle)
      // If no switch and descending, allow loop to exit
       if (switchcount === 0 && dir === "asc") {
           // This part might not be needed if we toggle direction at the start
           // dir = "desc";
           // switching = true;
       }
    }
  }
    // Optional: Add visual indicators for sorted column/direction
    var headers = table.rows[0].getElementsByTagName("TH");
    for (var j = 0; j < headers.length; j++) {
         headers[j].innerHTML = headers[j].innerHTML.replace(/ (↑|↓)$/, ""); // Remove old arrows
         if (j === n) {
             headers[j].innerHTML += (dir === "asc" ? " ↑" : " ↓");
         }
    }
}

// Initial sort by score (column 3) ascending on load
document.addEventListener('DOMContentLoaded', function() {
    var scoreHeader = document.getElementById("recipesTable").rows[0].getElementsByTagName("TH")[3];
    scoreHeader.setAttribute("data-sort-dir", "desc"); // Set initial direction so first click sorts asc
    sortTable(3); // Sort by score column initially
});
</script>

</body>
</html>
""")
        logging.info(f"Generated {html_index_path}")
    except IOError as e:
        logging.error(f"Failed to write {html_index_path}: {e}")
    except Exception as e:
         logging.error(f"Unexpected error generating HTML index: {e}")


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
                # Use .get() for safer access
                score = data.get("meal_info", {}).get("optimization_score", float('inf'))
                recipes.append({
                    "filename_base": os.path.splitext(filename)[0],
                    "score": score,
                    "json_filepath": filepath
                })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logging.warning(f"Error processing {filename} during cleanup: {e}")
                continue
            except Exception as e:
                 logging.warning(f"Unexpected error processing {filename} during cleanup: {e}")
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
            # Optional logging for debugging which files are marked
            # reason = f"score {recipe['score']:.2f} > {max_score}" if recipe["score"] > max_score else f"exceeds file limit ({kept_count+1} > {max_files})"
            # logging.debug(f"Marking {recipe['filename_base']} for removal ({reason})")


    # Remove marked files
    for recipe in files_to_remove:
        try:
            # Remove JSON
            if os.path.exists(recipe["json_filepath"]):
                 os.remove(recipe["json_filepath"])

            # Remove corresponding HTML
            html_file = os.path.join(html_dir, f"{recipe['filename_base']}.html")
            if os.path.exists(html_file):
                os.remove(html_file)

            removed_count += 1
            # Optional: Log removal
            # logging.info(f"Removed recipe files for {recipe['filename_base']}")
        except OSError as e:
            logging.error(f"Error removing files for {recipe['filename_base']}: {e}")
        except Exception as e:
             logging.error(f"Unexpected error removing files for {recipe['filename_base']}: {e}")


    remaining_count = len(recipes) - removed_count
    logging.info(f"Cleanup complete: Removed {removed_count} recipes. {remaining_count} remaining.")
    if recipes:
        try:
            best_score = recipes[0]['score'] if remaining_count > 0 else recipes[0]['score'] # Show best overall even if removed
            worst_remaining_score = recipes[min(remaining_count - 1, len(recipes)-1)]['score'] if remaining_count > 0 else 'N/A'
            logging.info(f"Best score overall: {best_score:.2f}")
            if worst_remaining_score != 'N/A':
                logging.info(f"Worst score remaining: {worst_remaining_score:.2f}")
        except IndexError:
             logging.info("No recipes found to determine best/worst scores.")
        except Exception as e:
             logging.warning(f"Could not determine best/worst scores after cleanup: {e}")


# --- Flask Routes and SocketIO Event Handlers ---

# Serve static files from recipes/html (for reports)
@app.route('/recipes/html/<path:filename>')
def serve_recipe_report(filename):
    # Use absolute path for safety or configure static folder properly
    # For simplicity, using relative path assuming script runs from project root
    report_dir = os.path.join(os.getcwd(), 'recipes', 'html')
    # Basic security check (optional but recommended)
    if not filename.endswith('.html'):
         return "Invalid file type", 404
    try:
        # Use send_from_directory for safer file serving
        return send_from_directory(report_dir, filename)
    except FileNotFoundError:
         return "Report not found", 404

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

        if not (1 <= num_foods <= 150): # Allow more foods?
            raise ValueError("Number of foods must be between 1 and 150.")
        if not (10 <= generations <= 2000): # Allow more generations
             raise ValueError("Generations must be between 10 and 2000.")
        if not (10 <= population_size <= 500): # Allow larger population
             raise ValueError("Population size must be between 10 and 500.")

    except (TypeError, ValueError) as e:
        logging.error(f"Invalid input data from {sid}: {e}")
        emit('optimization_error', {'message': f"Invalid input: {e}"}, room=sid)
        return # Stop processing

    # --- Ensure Data is Loaded ---
    if loaded_data["df"] is None or loaded_data["nutrient_mapping"] is None or loaded_data["rdi_values"] is None:
        logging.error("Server data (Excel/RDI) not loaded. Cannot start optimization.")
        emit('optimization_error', {'message': 'Server error: Core data files not loaded.'}, room=sid)
        return

    # --- Prepare Data for this Run ---
    df = loaded_data["df"]
    nutrient_mapping = loaded_data["nutrient_mapping"]
    rdi_values = loaded_data["rdi_values"]
    current_run_number = get_next_run_number() # Get run number here

    working_df = df.copy() # Work on a copy

    # Filter by diet type
    if diet_type != 'all':
        config_file = os.path.join('diets', f'{diet_type}.json')
        try:
            # Check if file exists before opening
            if not os.path.exists(config_file):
                 raise FileNotFoundError(f"Diet configuration file '{config_file}' not found.")

            with open(config_file, 'r', encoding='utf-8') as f:
                diet_config = json.load(f)

            initial_food_count = len(working_df)
            emit('status_update', {'message': f'Applying "{diet_type}" diet filter ({initial_food_count} foods)...'}, room=sid)

            # Apply inclusion first
            included_terms = diet_config.get('included_terms', [])
            if included_terms:
                included_terms_lower = [str(term).lower() for term in included_terms]
                # Ensure 'Food Name' column exists and handle potential NaN values
                if 'Food Name' in working_df.columns:
                    inclusion_mask = working_df['Food Name'].fillna('').astype(str).str.lower().apply(
                        lambda x: any(term in x for term in included_terms_lower)
                    )
                    working_df = working_df[inclusion_mask]
                    logging.info(f"Run #{current_run_number}: Included {len(working_df)} foods after inclusion filter.")
                else:
                     logging.warning(f"Run #{current_run_number}: 'Food Name' column not found for inclusion filter.")


            # Then apply exclusion
            excluded_terms = diet_config.get('excluded_terms', [])
            if excluded_terms:
                excluded_terms_lower = [str(term).lower() for term in excluded_terms]
                # Ensure 'Food Name' column exists and handle potential NaN values
                if 'Food Name' in working_df.columns:
                    exclusion_mask = ~working_df['Food Name'].fillna('').astype(str).str.lower().apply(
                        lambda x: any(term in x for term in excluded_terms_lower)
                    )
                    working_df = working_df[exclusion_mask]
                    logging.info(f"Run #{current_run_number}: {len(working_df)} foods remaining after exclusion filter.")
                else:
                     logging.warning(f"Run #{current_run_number}: 'Food Name' column not found for exclusion filter.")


            emit('status_update', {'message': f'Filtered to {len(working_df)} foods for "{diet_type}".'}, room=sid)

            if len(working_df) == 0:
                logging.warning(f"Run #{current_run_number}: No foods remain after filtering for {diet_type} diet")
                emit('optimization_error', {'message': f'No foods found matching the "{diet_type}" filter.'}, room=sid)
                return
        except FileNotFoundError as e:
            logging.error(f"Run #{current_run_number}: {e}")
            emit('optimization_error', {'message': str(e)}, room=sid)
            return
        except json.JSONDecodeError as e:
            logging.error(f"Run #{current_run_number}: Error decoding JSON from {config_file}: {e}")
            emit('optimization_error', {'message': f'Error reading diet file for "{diet_type}".'}, room=sid)
            return
        except Exception as e:
            logging.error(f"Run #{current_run_number}: Error applying diet filter {diet_type}: {e}")
            emit('optimization_error', {'message': f'Error processing diet filter: {e}'}, room=sid)
            return

    # Select random subset of foods
    if len(working_df) < num_foods:
        logging.warning(f"Run #{current_run_number}: Requested {num_foods} but only {len(working_df)} available after filtering. Using all available.")
        emit('status_update', {'message': f'Warning: Only {len(working_df)} foods available after filtering. Using all.'}, room=sid)
        num_foods_to_sample = len(working_df)
    else:
        num_foods_to_sample = num_foods

    if num_foods_to_sample <= 0:
         logging.error(f"Run #{current_run_number}: Invalid number of foods to sample ({num_foods_to_sample}).")
         emit('optimization_error', {'message': f'Cannot select {num_foods_to_sample} foods.'}, room=sid)
         return

    try:
         # Ensure sampling is possible
        if len(working_df) < num_foods_to_sample:
             raise ValueError(f"Cannot sample {num_foods_to_sample} foods, only {len(working_df)} available.")
        random_foods_df = working_df.sample(n=num_foods_to_sample)
        # emit('status_update', {'message': f'Selected {len(random_foods_df)} foods. Starting optimization...'}, room=sid)
    except ValueError as e:
         logging.error(f"Run #{current_run_number}: Error sampling foods: {e}")
         emit('optimization_error', {'message': f"Error selecting foods: {e}"}, room=sid)
         return
    except Exception as e: # Catch other potential sampling errors
         logging.error(f"Run #{current_run_number}: Unexpected error during food sampling: {e}")
         emit('optimization_error', {'message': f"Unexpected error selecting foods."}, room=sid)
         return

    # --- Start Optimization in Background Task ---
    task_args = (
        random_foods_df,
        nutrient_mapping,
        rdi_values,
        number_of_meals,
        meal_number,
        0.3, # randomness_factor - placeholder
        population_size,
        generations,
        diet_type,
        current_run_number, # Pass the run number determined here
        socketio, # Pass the instance
        sid       # Pass the specific client's SID
    )
    try:
        socketio.start_background_task(optimize_nutrition, *task_args)
        logging.info(f"Started background optimization task for SID: {sid}, Run #{current_run_number}")
    except Exception as e:
        logging.error(f"Run #{current_run_number}: Failed to start background task: {e}")
        emit('optimization_error', {'message': 'Server error: Could not start optimization task.'}, room=sid)


# --- Main Execution Block ---

def run_cli(args):
    """Runs the optimization in Command Line Interface mode."""
    logging.info("Running in CLI mode.")
    global_start_time = time.time()

    # --- Load Data ---
    df, nutrient_mapping, rdi_values = _load_data()
    if df is None or nutrient_mapping is None or rdi_values is None:
        logging.error("Failed to load necessary data (Excel/RDI). Exiting CLI mode.")
        return

    # --- Settings ---
    num_meals_cli = 1
    meal_num_cli = 1
    pop_size_cli = 100 # Example fixed value for CLI

    # diet_types_to_run = ['all', 'vegan', 'wfpb', 'nutrient_dense']
    diet_types_to_run = ['nutrient_dense'] # Quick test

    for diet_type in diet_types_to_run:
        run_start_time = time.time()
        current_run_number = get_next_run_number() # Get run number for this iteration
        logging.info(f"\n=== Starting CLI Optimization #{current_run_number}: {diet_type.upper()} ===")

        working_df = df.copy()

        # Filter foods based on diet type (similar logic to web handler, simplified logging)
        if diet_type != 'all':
            config_file = os.path.join('diets', f'{diet_type}.json')
            try:
                if not os.path.exists(config_file): raise FileNotFoundError(f"Diet file not found: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f: diet_config = json.load(f)
                initial_count = len(working_df)
                logging.info(f"Applying '{diet_type}' filter (Initial: {initial_count} foods)...")

                included_terms = diet_config.get('included_terms', [])
                if included_terms and 'Food Name' in working_df.columns:
                    included_terms_lower = [str(term).lower() for term in included_terms]
                    inclusion_mask = working_df['Food Name'].fillna('').astype(str).str.lower().apply(lambda x: any(term in x for term in included_terms_lower))
                    working_df = working_df[inclusion_mask]

                excluded_terms = diet_config.get('excluded_terms', [])
                if excluded_terms and 'Food Name' in working_df.columns:
                    excluded_terms_lower = [str(term).lower() for term in excluded_terms]
                    exclusion_mask = ~working_df['Food Name'].fillna('').astype(str).str.lower().apply(lambda x: any(term in x for term in excluded_terms_lower))
                    working_df = working_df[exclusion_mask]

                logging.info(f"Filtered to {len(working_df)} foods for '{diet_type}'.")
                if len(working_df) == 0:
                    logging.warning(f"No foods remain after filtering for {diet_type}. Skipping.")
                    continue
            except Exception as e:
                 logging.error(f"Error filtering for {diet_type}: {e}. Skipping.")
                 continue

        # Determine parameters for this run
        n_foods_cli = args.foods if args.foods is not None else random.randint(20, 40) # CLI specific range
        generations_cli = args.generations if args.generations is not None else random.randint(150, 400) # CLI specific range
        # generations_cli = 5 # Force quick test

        num_available = len(working_df)
        if num_available < n_foods_cli:
            logging.warning(f"Requested {n_foods_cli} foods, but only {num_available} available for {diet_type}. Using all.")
            n_foods_to_sample = num_available
        else:
            n_foods_to_sample = n_foods_cli

        if n_foods_to_sample <= 0:
             logging.error(f"Cannot run optimization with {n_foods_to_sample} foods. Skipping {diet_type}.")
             continue

        try:
             random_foods_df_cli = working_df.sample(n=n_foods_to_sample)
        except ValueError as e:
             logging.error(f"Error sampling {n_foods_to_sample} foods for CLI run: {e}. Skipping.")
             continue

        logging.info(f"Selected {len(random_foods_df_cli)} foods. Generations: {generations_cli}. Population: {pop_size_cli}.")

        # --- Run optimization (CLI mode: no socketio instance) ---
        result = optimize_nutrition(
            food_df=random_foods_df_cli,
            nutrient_mapping=nutrient_mapping,
            rdi_targets=rdi_values,
            number_of_meals=num_meals_cli,
            meal_number=meal_num_cli,
            population_size=pop_size_cli,
            generations=generations_cli,
            diet_type=diet_type,
            run_number=current_run_number,
            socketio_instance=None,
            sid=None
        )

        # --- Process CLI Result ---
        if result and result.get("solution"):
            # Recreate `available_foods` dict for the print function if needed
            # This requires the nutrient_mapping and the specific random_foods_df used
            cli_available_foods_print = {}
            nutrient_cols_cli = [col for col in nutrient_mapping.keys() if col in random_foods_df_cli.columns]
            for idx, row in random_foods_df_cli.iterrows():
                 food_name_cli = row.get('Food Name')
                 if isinstance(food_name_cli, str):
                     food_data_cli = {'density': 100}
                     for nutrient_col in nutrient_cols_cli:
                         val_cli = pd.to_numeric(row.get(nutrient_col), errors='coerce')
                         food_data_cli[nutrient_col] = 0.0 if pd.isna(val_cli) else float(val_cli)
                     cli_available_foods_print[food_name_cli.strip()] = food_data_cli

            # Print console report using the final solution and recreated food data
            print_nutrition_report(
                 result["solution"],
                 cli_available_foods_print, # Use the recreated dict
                 rdi_values,
                 nutrient_mapping,
                 num_meals_cli,
                 meal_num_cli
            )
            logging.info(f"=== Completed CLI Optimization #{current_run_number}: {diet_type.upper()} (Score: {result['score']:.2f}, Time: {time.time() - run_start_time:.2f}s) ===\n")
        else:
            logging.error(f"Optimization #{current_run_number} failed for {diet_type.upper()}.")
            logging.info(f"=== Aborted CLI Optimization #{current_run_number}: {diet_type.upper()} ===\n")


    # --- Post-run tasks for CLI ---
    logging.info("CLI run(s) finished. Performing cleanup and index generation.")
    cleanup_high_score_recipes(max_score=25, max_files=300) # Example cleanup params
    generate_index()
    logging.info(f"Total CLI execution time: {time.time() - global_start_time:.2f} seconds")


def run_webui():
    """Loads data and starts the Flask-SocketIO web server."""
    logging.info("Running in Web UI mode.")

    # --- Load Data Globally for Flask App ---
    df, nutrient_mapping, rdi_values = _load_data()
    if df is None or nutrient_mapping is None or rdi_values is None:
        logging.error("CRITICAL: Failed to load data for Web UI (Excel/RDI). Server may not function correctly.")
        # Allow server to start but log critical error
    else:
        loaded_data["df"] = df
        loaded_data["nutrient_mapping"] = nutrient_mapping
        loaded_data["rdi_values"] = rdi_values
        logging.info("Data loaded successfully for Web UI.")

    # --- Start Server ---
    host_ip = '0.0.0.0' # Listen on all interfaces
    port_num = 5000
    logging.info(f"Starting Flask-SocketIO server on http://{host_ip}:{port_num} ...")
    try:
        # Use eventlet recommended by Flask-SocketIO for async operations
        # Add use_reloader=False to prevent issues with background tasks in debug mode if enabled
        socketio.run(app, host=host_ip, port=port_num, debug=False, use_reloader=False)
    except ImportError:
         logging.error("Eventlet not installed. Falling back to Werkzeug development server (may have limited concurrency).")
         logging.error("Install eventlet: pip install eventlet")
         # Fallback without eventlet (less ideal for SocketIO)
         app.run(host=host_ip, port=port_num, debug=False)
    except Exception as e:
        logging.error(f"Failed to start server: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize nutrition using genetic algorithm (CLI or Web UI)')
    parser.add_argument('--webui', action='store_true', help='Run as a Flask web UI instead of CLI')
    parser.add_argument('--generations', type=int, default=None, help='(CLI Mode) Number of generations (default: random)')
    parser.add_argument('--foods', type=int, default=None, help='(CLI Mode) Number of foods to select (default: random)')
    args = parser.parse_args()

    init_directories() # Ensure directories exist regardless of mode

    if args.webui:
        run_webui()
    else:
        run_cli(args)