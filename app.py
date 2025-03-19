from datetime import datetime
import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, logger=True, engineio_logger=True)

# Global variables for optimization state
optimization_running = False
optimization_thread = None
should_stop = False

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Load food data globally
logger.debug("Loading food data from Excel and RDI JSON")
file_path = "Release 2 - Nutrient file.xlsx"
try:
    df = pd.read_excel(file_path, sheet_name="All solids & liquids per 100g")
    logger.debug(f"Loaded {len(df)} foods from Excel file")
except Exception as e:
    logger.error(f"Failed to load Excel file: {e}")
    raise

rdi_path = os.path.join('rdi', 'rdi.json')
try:
    with open(rdi_path, 'r') as f:
        nutrient_mapping = json.load(f)
    rdi_values = {nutrient: details['rdi'] for nutrient, details in nutrient_mapping.items()}
    logger.debug(f"Loaded {len(rdi_values)} nutrients from RDI JSON")
except Exception as e:
    logger.error(f"Failed to load RDI JSON: {e}")
    raise

def get_next_run_number():
    if not os.path.exists('optimization_history.csv'):
        logger.debug("No optimization history file found, starting with run number 1")
        return 1
    df = pd.read_csv('optimization_history.csv')
    run_number = df['run_number'].max() + 1 if not df.empty else 1
    logger.debug(f"Next run number: {run_number}")
    return run_number

def optimize_nutrition_core(food_df, nutrient_mapping, rdi_targets, params):
    global optimization_running, should_stop

    number_of_meals = params.get('number_of_meals', 1)
    meal_number = params.get('meal_number', 1)
    population_size = params.get('population_size', 50)
    generations = params.get('generations', 100)
    diet_type = params.get('diet_type', 'all')
    randomness_factor = params.get('randomness_factor', 0.3)

    logger.debug(f"Params: number_of_meals={number_of_meals}, meal_number={meal_number}, "
                 f"population_size={population_size}, generations={generations}, diet_type={diet_type}, "
                 f"randomness_factor={randomness_factor}")

    if meal_number < 1 or meal_number > number_of_meals:
        logger.error(f"Invalid meal number: {meal_number} not between 1 and {number_of_meals}")
        raise ValueError(f"Meal number must be between 1 and {number_of_meals}")

    meal_rdi_targets = {nutrient: target / number_of_meals for nutrient, target in rdi_targets.items()}
    available_foods = {}

    logger.debug(f"Processing {len(food_df)} foods into available_foods")
    for idx, row in food_df.iterrows():
        food_name = row['Food Name']
        food_data = {'density': 100}
        for column_name in nutrient_mapping.keys():
            if column_name in food_df.columns:
                food_data[column_name] = row[column_name] if pd.notna(row[column_name]) else 0.0
            else:
                food_data[column_name] = 0.0
        available_foods[food_name] = food_data

    foods_df = pd.DataFrame(available_foods).T
    nutrient_cols = [col for col in foods_df.columns if col != 'density']
    foods_df['nutrient_score'] = 0
    for nutrient in nutrient_cols:
        if nutrient in meal_rdi_targets:
            weight = 1 / meal_rdi_targets[nutrient] if meal_rdi_targets[nutrient] > 0 else 0
            foods_df['nutrient_score'] += foods_df[nutrient] * weight / foods_df['density']

    def create_solution():
        return {food: random.uniform(25, 100) for food in available_foods}

    penalties = {
        "under_rdi": random.uniform(2.0, 3.0),
        "over_rdi": random.uniform(0.2, 0.4),
        "over_ul": random.uniform(1.8, 2.5),
        "water_soluble": random.uniform(0.1, 0.3),
        "fat_soluble": random.uniform(0.3, 0.5)
    }
    logger.debug(f"Penalties: {penalties}")

    def evaluate_solution(solution):
        current_nutrients = {nutrient: 0 for nutrient in meal_rdi_targets}
        for food, amount in solution.items():
            for nutrient in meal_rdi_targets:
                if nutrient in foods_df.loc[food]:
                    nutrient_per_gram = foods_df.loc[food][nutrient] / foods_df.loc[food]['density']
                    current_nutrients[nutrient] += nutrient_per_gram * amount

        score = _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties)

        # Debug information for nutrition calculation
        if logger.level <= logging.DEBUG:
            # Log top 3 contributors to score
            nutrition_penalties = []
            for nutrient, amount in current_nutrients.items():
                target = meal_rdi_targets.get(nutrient, 0)
                if target > 0:
                    if amount < target:
                        penalty = (target - amount) / target * penalties["under_rdi"]
                        nutrition_penalties.append((nutrient, "under", penalty, amount, target))
                    elif amount > target * 2:  # Over 200%
                        # Check if it's water or fat soluble
                        soluble_type = "water_soluble" if nutrient in ["vitamin_c", "vitamin_b1", "vitamin_b2", "vitamin_b3", "vitamin_b5", "vitamin_b6", "vitamin_b9", "vitamin_b12"] else "fat_soluble"
                        excess_ratio = (amount - target * 2) / target
                        penalty = excess_ratio * penalties[soluble_type]
                        nutrition_penalties.append((nutrient, "excess", penalty, amount, target))
                    elif amount > target:
                        penalty = (amount - target) / target * penalties["over_rdi"]
                        nutrition_penalties.append((nutrient, "over", penalty, amount, target))

            # Sort by penalty and log top contributors
            nutrition_penalties.sort(key=lambda x: x[2], reverse=True)
            top_penalties = nutrition_penalties[:3]
            logger.debug(f"Top penalties: {top_penalties}")

        return score

    def mutate_solution(solution):
        # Create a copy to avoid modifying the original
        mutated = solution.copy()

        # Debug: log before mutation values for a few keys
        if logger.level <= logging.DEBUG:
            sample_foods = list(mutated.keys())[:3]
            before_values = {food: mutated[food] for food in sample_foods}
            logger.debug(f"Before mutation (sample): {before_values}")

        # Apply mutation to multiple foods
        for food in mutated.keys():
            # Increase mutation probability for more diversity
            if random.random() < randomness_factor:
                # Use a wider range for mutation
                change = random.uniform(-25, 25)
                mutated[food] = max(0, mutated[food] + change)

        # Debug: log after mutation values
        if logger.level <= logging.DEBUG:
            after_values = {food: mutated[food] for food in sample_foods}
            logger.debug(f"After mutation (sample): {after_values}")

        return mutated

    def crossover_solution(sol1, sol2):
        new_solution = {}

        # Debug: log parents for a few keys
        if logger.level <= logging.DEBUG:
            sample_foods = list(sol1.keys())[:3]
            parent1_values = {food: sol1[food] for food in sample_foods}
            parent2_values = {food: sol2[food] for food in sample_foods}
            logger.debug(f"Crossover parents (sample): {parent1_values} x {parent2_values}")

        for food in sol1:
            # Use weighted average crossover for more exploration
            alpha = random.random()
            new_solution[food] = alpha * sol1[food] + (1 - alpha) * sol2[food]

        # Debug: log result for a few keys
        if logger.level <= logging.DEBUG:
            child_values = {food: new_solution[food] for food in sample_foods}
            logger.debug(f"Crossover child (sample): {child_values}")

        return new_solution

    # Initialize population
    population = [create_solution() for _ in range(population_size)]
    best_solution = None
    best_score = float('inf')
    run_number = get_next_run_number()

    socketio.emit('status_update', {'message': f'Starting optimization run #{run_number}'})
    logger.info(f"Starting optimization run #{run_number}")
    print(f"Starting optimization run #{run_number}")

    # Log initial population stats to have a baseline
    initial_scores = [evaluate_solution(sol) for sol in population]
    logger.debug(f"Initial population score stats: min={min(initial_scores):.2f}, max={max(initial_scores):.2f}, avg={sum(initial_scores)/len(initial_scores):.2f}")

    start_time = time.time()
    for generation in range(generations):
        if should_stop:
            socketio.emit('status_update', {'message': 'Optimization stopped by user'})
            logger.info("Optimization stopped by user")
            print("Optimization stopped by user")
            optimization_running = False
            return None

        # Evaluate population
        scores = [(evaluate_solution(sol), sol) for sol in population]
        scores.sort(key=lambda x: x[0])
        current_best_score, current_best_solution = scores[0]

        # Log population diversity metrics
        all_scores = [score for score, _ in scores]
        logger.debug(f"Gen {generation+1} score stats: min={min(all_scores):.2f}, max={max(all_scores):.2f}, avg={sum(all_scores)/len(all_scores):.2f}, std={np.std(all_scores):.2f}")

        # Update best solution if improved
        if current_best_score < best_score:
            best_score = current_best_score
            best_solution = current_best_solution.copy()
            logger.debug(f"New best score at generation {generation + 1}: {best_score:.2f}")

            # Log the best solution (sample)
            sample_solution = {k: best_solution[k] for k in list(best_solution.keys())[:5] if best_solution[k] > 10}
            logger.debug(f"Sample of best solution: {sample_solution}")
        else:
            # Log when score is not improving
            logger.debug(f"No improvement at generation {generation + 1}. Current best: {current_best_score:.2f}, Overall best: {best_score:.2f}")

        # Save generation data
        generation_data = pd.DataFrame([{
            'run_number': run_number,
            'generation': generation + 1,
            'score': best_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        if os.path.exists('optimization_history.csv'):
            generation_data.to_csv('optimization_history.csv', mode='a', header=False, index=False)
        else:
            generation_data.to_csv('optimization_history.csv', index=False)

        # Emit update to UI
        nutrition_report = generate_nutrition_report(best_solution, available_foods, rdi_targets, number_of_meals, meal_number)
        socketio.emit('optimization_update', {
            'generation': generation + 1,
            'total_generations': generations,
            'best_score': round(best_score, 2),
            'nutrition_report': nutrition_report
        })
        logger.debug(f"Generation {generation + 1}/{generations}, Best Score: {best_score:.2f}")
        print(f"Generation {generation + 1}/{generations}, Best Score: {best_score:.2f}")

        time.sleep(0.1)  # Ensure UI updates

        # Tournament selection instead of just selecting the top performers
        new_population = []
        # Keep some elite solutions (top performers)
        elite_size = max(1, int(population_size * 0.1))  # Keep top 10%
        elite = [sol for _, sol in scores[:elite_size]]
        new_population.extend(elite)
        logger.debug(f"Kept {len(elite)} elite solutions")

        # Count the number of crossovers and mutations performed
        crossovers = 0
        mutations = 0

        # Use tournament selection for the rest
        while len(new_population) < population_size:
            # Tournament selection
            candidates = random.sample(population, min(5, len(population)))
            candidate_scores = [(evaluate_solution(c), c) for c in candidates]
            candidate_scores.sort(key=lambda x: x[0])
            winner = candidate_scores[0][1]

            # Add mutation and crossover with probability
            if len(new_population) >= 2 and random.random() < 0.7:
                # Crossover
                parent2 = random.choice(new_population)
                child = crossover_solution(winner, parent2)
                crossovers += 1

                # Mutation with higher probability
                if random.random() < 0.4:
                    child = mutate_solution(child)
                    mutations += 1
                new_population.append(child)
            else:
                # Just mutation
                mutated = mutate_solution(winner)
                mutations += 1
                new_population.append(mutated)

        logger.debug(f"Generation {generation+1}: performed {crossovers} crossovers and {mutations} mutations")

        # Evaluate the new population to get diversity stats
        new_scores = [evaluate_solution(sol) for sol in new_population]
        logger.debug(f"New population score stats: min={min(new_scores):.2f}, max={max(new_scores):.2f}, avg={sum(new_scores)/len(new_scores):.2f}, std={np.std(new_scores):.2f}")

        # Ensure we have exactly population_size solutions
        population = new_population[:population_size]

    execution_time = time.time() - start_time
    final_report = save_nutrition_report(
        best_solution, available_foods, rdi_targets, best_score, run_number,
        number_of_meals, meal_number, generations, execution_time, diet_type, penalties
    )

    final_report_data = json.load(open(final_report))
    socketio.emit('optimization_complete', {
        'final_report': {
            'meal_info': final_report_data['meal_info'],
            'foods': {food: float(amount[:-1]) for food, amount in final_report_data['food_quantities'].items()},
            'nutrients': [
                {
                    'name': nutrient,
                    'amount': float(''.join(c for c in info['amount'] if c.isdigit() or c == '.')),
                    'unit': _get_unit(nutrient),
                    'meal_percentage': float(info['meal_percentage'].rstrip('%')),
                    'daily_percentage': float(info['daily_percentage'].rstrip('%')),
                    'status': info['status']
                } for nutrient, info in final_report_data['nutrition_profile'].items()
            ],
            'summary': final_report_data['summary']
        }
    })
    logger.info(f"Optimization complete, execution time: {execution_time:.2f}s, report saved to: {final_report}")
    print(f"Optimization complete, execution time: {execution_time:.2f}s, report saved to: {final_report}")

    optimization_running = False
    return {
        "solution": {food: round(amount) for food, amount in best_solution.items() if amount > 10},
        "penalties": penalties
    }

def generate_nutrition_report(solution, food_data, rdi, number_of_meals, meal_number):
    meal_rdi = {nutrient: target / number_of_meals for nutrient, target in rdi.items()}
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}

    for food, amount in solution.items():
        for nutrient in meal_rdi:
            if nutrient in food_data[food]:
                nutrient_per_gram = food_data[food][nutrient] / food_data[food]['density']
                final_nutrients[nutrient] += nutrient_per_gram * amount

    report = {
        'meal_info': {
            'meal_number': meal_number,
            'number_of_meals': number_of_meals,
            'target_percentage': 100 / number_of_meals
        },
        'foods': {food: round(amount) for food, amount in solution.items() if amount > 10},
        'nutrients': [
            {
                'name': nutrient,
                'amount': round(amount, 1),
                'unit': _get_unit(nutrient),
                'meal_percentage': round(amount / meal_rdi[nutrient] * 100, 1),
                'daily_percentage': round(amount / rdi[nutrient] * 100, 1),
                'status': "LOW" if amount/meal_rdi[nutrient]*100 < 80 else "OK" if amount/meal_rdi[nutrient]*100 < 150 else "HIGH"
            } for nutrient, amount in final_nutrients.items()
        ],
        'summary': {
            'ok': sum(1 for n, a in final_nutrients.items() if 80 <= a/meal_rdi[n]*100 < 150),
            'low': sum(1 for n, a in final_nutrients.items() if a/meal_rdi[n]*100 < 80),
            'high': sum(1 for n, a in final_nutrients.items() if a/meal_rdi[n]*100 >= 150),
            'total': len(meal_rdi)
        }
    }
    return report

@app.route('/')
def index():
    foods = {row['Food Name']: True for _, row in df.iterrows()}
    logger.debug(f"Rendering index with {len(foods)} foods")
    return render_template('index.html', foods=foods)

@app.route('/start_optimization', methods=['POST'])
def start_optimization():
    global optimization_running, optimization_thread, should_stop

    logger.debug("Received start_optimization request")
    if optimization_running:
        logger.warning("Optimization already running")
        return jsonify({'status': 'error', 'message': 'Optimization already running'})

    data = request.get_json()
    logger.debug(f"Request data: {data}")
    selected_foods = data.get('selected_foods', [])
    food_df = df[df['Food Name'].isin(selected_foods)]

    if len(food_df) == 0:
        logger.error("No foods selected")
        return jsonify({'status': 'error', 'message': 'No foods selected'})

    params = {
        'number_of_meals': data.get('number_of_meals', 1),
        'meal_number': data.get('meal_number', 1),
        'population_size': data.get('population_size', 50),
        'generations': data.get('generations', 100),
        'diet_type': 'all',
        'randomness_factor': 0.3
    }
    logger.debug(f"Optimization params: {params}")

    should_stop = False
    optimization_running = True
    optimization_thread = threading.Thread(
        target=optimize_nutrition_core,
        args=(food_df, nutrient_mapping, rdi_values, params)
    )
    optimization_thread.daemon = True
    logger.debug("Starting optimization thread")
    optimization_thread.start()
    logger.info("Optimization thread started successfully")
    return jsonify({'status': 'success', 'message': 'Optimization started'})

@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    global should_stop, optimization_running, optimization_thread

    logger.debug("Received stop_optimization request")
    if optimization_running:
        should_stop = True
        optimization_thread.join()
        optimization_running = False
        logger.info("Optimization stopped successfully")
        return jsonify({'status': 'success', 'message': 'Optimization stopped'})
    logger.warning("No optimization running to stop")
    return jsonify({'status': 'error', 'message': 'No optimization running'})

def _calculate_nutrition_score(current, targets, penalties):
    score = 0
    energy_key = "Energy with dietary fibre, equated (kJ)"
    if energy_key in current and energy_key in targets:
        energy = current[energy_key]
        energy_target = targets[energy_key]
        with open(rdi_path, 'r') as f:
            nutrient_data = json.load(f)
            rdi = nutrient_data[energy_key]['rdi']
            ul = nutrient_data[energy_key]['ul']
        if energy < rdi:
            score += ((rdi - energy) / rdi) ** 2 * penalties["under_rdi"]
        elif energy > ul:
            score += ((energy - ul) / ul) ** 2 * penalties["over_ul"]

    water_soluble_vitamins = [
        'Vitamin C (mg)', 'Thiamin (B1) (mg)', 'Riboflavin (B2) (mg)', 'Niacin (B3) (mg)',
        'Pantothenic acid (B5) (mg)', 'Pyridoxine (B6) (mg)', 'Biotin (B7) (ug)',
        'Cobalamin (B12) (ug)', 'Total folates (ug)'
    ]
    fat_soluble_vitamins = [
        'Vitamin A retinol equivalents (ug)', 'Vitamin D3 equivalents (ug)', 'Vitamin E (mg)'
    ]

    for nutrient, target in targets.items():
        if nutrient == energy_key:
            continue
        current_value = current.get(nutrient, 0)
        if current_value < target:
            score += ((target - current_value) / target) ** 2 * penalties["under_rdi"]
        else:
            if nutrient in water_soluble_vitamins:
                penalty = penalties["water_soluble"]
            elif nutrient in fat_soluble_vitamins:
                penalty = penalties["fat_soluble"]
            else:
                penalty = penalties["over_rdi"]
            score += ((current_value - target) / target) ** 2 * penalty
    return score

def save_nutrition_report(foods_consumed, food_data, rdi, score, run_number, number_of_meals=3, meal_number=1, generations=100, execution_time=0, diet_type='all', penalties=None):
    meal_rdi = {nutrient: float(target) / number_of_meals for nutrient, target in rdi.items()}
    recipe_timestamp = datetime.now().strftime('%Y-%m-%d')
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}

    for food, amount in foods_consumed.items():
        for nutrient in meal_rdi:
            if nutrient in food_data[food]:
                nutrient_per_gram = float(food_data[food][nutrient]) / float(food_data[food]['density'])
                final_nutrients[nutrient] += nutrient_per_gram * amount

    report = {
        "meal_info": {
            "run_number": int(run_number),
            "diet_type": str(diet_type),
            "target_percentage": float(100/number_of_meals),
            "optimization_score": float(score),
            "generations": int(generations),
            "number_of_foods": len(foods_consumed),
            "execution_time_seconds": float(execution_time),
            "date": str(recipe_timestamp),
            "penalties": penalties
        },
        "food_quantities": {str(food): f"{float(amount):.1f}g" for food, amount in sorted(foods_consumed.items(), key=lambda x: float(x[1]), reverse=True)},
        "nutrition_profile": {
            str(nutrient): {
                "amount": f"{float(amount):.1f}{_get_unit(nutrient)}",
                "meal_percentage": f"{(float(amount)/float(meal_rdi[nutrient])*100):.1f}%",
                "daily_percentage": f"{(float(amount)/float(rdi[nutrient])*100):.1f}%",
                "status": "LOW" if float(amount)/float(meal_rdi[nutrient])*100 < 80 else "OK" if float(amount)/float(meal_rdi[nutrient])*100 < 150 else "HIGH"
            } for nutrient, amount in final_nutrients.items()
        },
        "summary": {
            "nutrients_at_good_levels": sum(1 for n, a in final_nutrients.items() if 80 <= float(a)/float(meal_rdi[n])*100 < 150),
            "nutrients_below_target": sum(1 for n, a in final_nutrients.items() if float(a)/float(meal_rdi[n])*100 < 80),
            "nutrients_above_target": sum(1 for n, a in final_nutrients.items() if float(a)/float(meal_rdi[n])*100 >= 150),
            "total_nutrients": len(meal_rdi)
        }
    }

    json_filename = os.path.join('recipes', 'json', f'meal_{run_number}_{timestamp}.json')
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return json_filename

def _get_unit(nutrient):
    if '(mg)' in nutrient:
        return 'mg'
    elif '(ug)' in nutrient:
        return 'μg'
    elif '(g)' in nutrient:
        return 'g'
    return ''

def init_directories():
    logger.debug("Initializing directories")
    os.makedirs('recipes/json', exist_ok=True)
    os.makedirs('recipes/html', exist_ok=True)
    os.makedirs('rdi', exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize nutrition using genetic algorithm')
    parser.add_argument('--mode', choices=['cli', 'ui'], default='cli', help='Run mode: cli or ui')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--foods', type=int, default=None)
    args = parser.parse_args()

    init_directories()

    if args.mode == 'ui':
        logger.info("Starting Flask app in UI mode")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    else:
        logger.info("Running in CLI mode")
        start_time = time.time()
        n_foods = args.foods if args.foods else random.randint(10, 30)
        random_foods = df.sample(n=n_foods)
        result = optimize_nutrition_core(
            random_foods,
            nutrient_mapping,
            rdi_values,
            {'number_of_meals': 1, 'meal_number': 1, 'population_size': 50, 'generations': args.generations, 'diet_type': 'all', 'randomness_factor': 0.3}
        )