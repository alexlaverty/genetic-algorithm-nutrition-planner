from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
import random
import time

# Save to JSON file with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def get_next_run_number():
    """Get the next run number by checking existing CSV file"""
    if not os.path.exists('optimization_history.csv'):
        return 1
    df = pd.read_csv('optimization_history.csv')
    return df['run_number'].max() + 1 if not df.empty else 1

def optimize_nutrition(food_df,
                       nutrient_mapping,
                       rdi_targets,
                       number_of_meals=1,
                       meal_number=1,
                       randomness_factor=0.3,
                       population_size=50,
                       generations=100,
                       diet_type='all'):
    """
    Optimize food selection to meet RDI targets while maintaining variety,
    scaled for a specific meal in a multi-meal day.

    Parameters:
    -----------
    food_df : pandas.DataFrame
        DataFrame containing food nutrition data from the Excel spreadsheet

    nutrient_mapping : dict
        Dictionary mapping from nutrient names to column names in the DataFrame
        Example: {'protein': 'Protein (g)', 'vitamin_c': 'Vitamin C (mg)', ...}

    rdi_targets : dict
        Dictionary with nutrient names as keys and RDI values as values.
        Example: {'protein': 50, 'vitamin_c': 90, ...}

    number_of_meals : int
        Number of meals per day to divide the RDI targets between

    meal_number : int
        Which meal of the day this is (1 to number_of_meals)

    num_iterations : int
        Number of optimization attempts to find the best solution

    randomness_factor : float
        Value between 0-1 determining how much randomness to include
        Higher values = more variety but potentially less optimal solutions

    population_size : int
        Number of solutions in the population for the genetic algorithm

    generations : int
        Number of generations to run the genetic algorithm

    Returns:
    --------
    dict
        Dictionary with food names as keys and grams to consume as values
    """
    # Validate inputs
    if meal_number < 1 or meal_number > number_of_meals:
        raise ValueError(f"Meal number must be between 1 and {number_of_meals}")

    # Scale RDI targets for a single meal
    meal_rdi_targets = {nutrient: target / number_of_meals for nutrient, target in rdi_targets.items()}

    # Convert Excel data to the format needed for optimization
    available_foods = {}

    for idx, row in food_df.iterrows():
        food_name = row['Food Name']
        food_data = {'density': 100}  # Assuming 100g per serving

        # Map each nutrient using the column names directly from nutrient_mapping
        for column_name in nutrient_mapping.keys():
            if column_name in food_df.columns:
                # Handle missing values
                if pd.notna(row[column_name]):
                    food_data[column_name] = row[column_name]
                else:
                    food_data[column_name] = 0.0
            else:
                print(f"Warning: Column '{column_name}' not found")
                food_data[column_name] = 0.0

        available_foods[food_name] = food_data

    # Convert to DataFrame for easier manipulation
    foods_df = pd.DataFrame(available_foods).T

    # Calculate nutrient density scores for each food
    # Higher score = more nutrients per gram
    nutrient_cols = [col for col in foods_df.columns if col != 'density']

    # Calculate weighted nutrient score based on RDI percentages
    foods_df['nutrient_score'] = 0
    for nutrient in nutrient_cols:
        if nutrient in meal_rdi_targets:
            # Weight by importance (inverse of RDI - smaller RDIs are more important per unit)
            weight = 1 / meal_rdi_targets[nutrient] if meal_rdi_targets[nutrient] > 0 else 0
            foods_df['nutrient_score'] += foods_df[nutrient] * weight / foods_df['density']

    def create_solution():
        """Create a random solution."""
        return {food: random.uniform(25, 100) for food in available_foods}

    # Generate random penalties for this optimization run
    penalties = {
        "under_rdi": random.uniform(1.3, 2.0),      # Penalty for being under RDI
        "over_rdi": random.uniform(0.3, 1.0),       # Base penalty for being over RDI
        "over_ul": random.uniform(2.0, 3.0),        # Severe penalty for exceeding UL
        "water_soluble": random.uniform(0.3, 0.7),  # More lenient for water-soluble vitamins
        "fat_soluble": random.uniform(0.6, 0.9)     # Less lenient for fat-soluble vitamins
    }

    print(f"Penalties:")
    print(penalties)

    def evaluate_solution(solution):
        """Evaluate how close the solution is to the RDI targets."""
        current_nutrients = {nutrient: 0 for nutrient in meal_rdi_targets}
        for food, amount in solution.items():
            for nutrient in meal_rdi_targets:
                if nutrient in foods_df.loc[food]:
                    nutrient_per_gram = foods_df.loc[food][nutrient] / foods_df.loc[food]['density']
                    current_nutrients[nutrient] += nutrient_per_gram * amount
        return _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties)


    def mutate_solution(solution):
        """Mutate the solution by tweaking the food amounts."""
        food = random.choice(list(solution.keys()))
        solution[food] = max(0, solution[food] + random.uniform(-20, 20))  # Increased mutation range
        return solution

    def crossover_solution(sol1, sol2):
        """Create a new solution by combining two solutions."""
        new_solution = {}
        for food in sol1:
            new_solution[food] = sol1[food] if random.random() < 0.5 else sol2[food]
        return new_solution

    # Initialize population
    population = [create_solution() for _ in range(population_size)]
    best_solution = None
    best_score = float('inf')

    # Get run number at the start
    run_number = get_next_run_number()
    print(f"Starting optimization run #{run_number}")

    for generation in range(generations):
        # Evaluate population
        scores = [(evaluate_solution(sol), sol) for sol in population]
        scores.sort(key=lambda x: x[0])
        best_score, best_solution = scores[0]

        # Save this generation's data immediately
        generation_data = pd.DataFrame([{
            'run_number': run_number,
            'generation': generation + 1,
            'score': best_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])

        # Append to CSV file (create if doesn't exist)
        if os.path.exists('optimization_history.csv'):
            generation_data.to_csv('optimization_history.csv', mode='a', header=False, index=False)
        else:
            generation_data.to_csv('optimization_history.csv', index=False)

        # Select the best solutions
        population = [sol for _, sol in scores[:population_size // 2]]

        # Create new population through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover_solution(parent1, parent2)
            if random.random() < 0.1:
                child = mutate_solution(child)
            new_population.append(child)
        population = new_population
        print('.', end='', flush=True)

    execution_time = time.time() - start_time  # Calculate total time
    print()
    print(f"Execution time: {execution_time:.2f} seconds")

    # Save the nutrition report instead of printing it
    report_file = save_nutrition_report(best_solution,
                                        available_foods,
                                        rdi_targets,
                                        best_score,
                                        run_number,  # Add run_number parameter
                                        number_of_meals,
                                        meal_number,
                                        generations=generations,
                                        execution_time=execution_time,
                                        diet_type=diet_type,
                                        penalties=penalties)

    print(f"Generation {generation + 1}/{generations}, Best Score: {best_score}")
    print(f"Report saved to: {report_file}")

    # Round the amounts to make them more practical
    return {
        "solution": {food: round(amount) for food, amount in best_solution.items() if amount > 10},
        "penalties": penalties
    }

def _is_nutrition_sufficient(current, targets, threshold=0.90):
    """Check if current nutrients are at least threshold percent of targets."""
    for nutrient, target in targets.items():
        if current.get(nutrient, 0) < threshold * target:
            return False
    return True

def _calculate_nutrition_score(current, targets, penalties):
    """Calculate how far current nutrients are from targets with randomized penalties."""
    score = 0

    # Special handling for energy/calories
    energy_key = "Energy with dietary fibre, equated (kJ)"
    if energy_key in current and energy_key in targets:
        energy = current[energy_key]
        energy_target = targets[energy_key]

        with open('rdi.json', 'r') as f:
            nutrient_data = json.load(f)
            rdi = nutrient_data[energy_key]['rdi']
            ul = nutrient_data[energy_key]['ul']

        if energy < rdi:
            score += ((rdi - energy) / rdi) ** 2 * penalties["under_rdi"]
        elif energy > ul:
            score += ((energy - ul) / ul) ** 2 * penalties["over_ul"]

    # Handle other nutrients
    water_soluble_vitamins = [
        'Vitamin C (mg)',
        'Thiamin (B1) (mg)',
        'Riboflavin (B2) (mg)',
        'Niacin (B3) (mg)',
        'Pantothenic acid (B5) (mg)',
        'Pyridoxine (B6) (mg)',
        'Biotin (B7) (ug)',
        'Cobalamin (B12) (ug)',
        'Total folates (ug)'
    ]

    fat_soluble_vitamins = [
        'Vitamin A retinol equivalents (ug)',
        'Vitamin D3 equivalents (ug)',
        'Vitamin E (mg)'
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

def save_nutrition_report(foods_consumed, food_data, rdi, score, run_number,
                         number_of_meals=3, meal_number=1, generations=100,
                         execution_time=0, diet_type='all', penalties=None):
    """Save a detailed nutrition report as JSON and HTML."""

    # Scale RDI for a single meal
    meal_rdi = {nutrient: float(target) / number_of_meals for nutrient, target in rdi.items()}

    # Calculate nutrient totals
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}

    for food, amount in foods_consumed.items():
        for nutrient in meal_rdi:
            if nutrient in food_data[food]:
                nutrient_per_gram = float(food_data[food][nutrient]) / float(food_data[food]['density'])
                final_nutrients[nutrient] += nutrient_per_gram * amount

    # Create report dictionary
    report = {
        "meal_info": {
            "run_number": int(run_number),
            "diet_type": str(diet_type),
            "target_percentage": float(100/number_of_meals),
            "optimization_score": float(score),
            "generations": int(generations),
            "number_of_foods": len(foods_consumed),
            "execution_time_seconds": float(execution_time),
            "penalties": {
                "under_rdi": penalties["under_rdi"],
                "over_rdi": penalties["over_rdi"],
                "over_ul": penalties["over_ul"],
                "water_soluble": penalties["water_soluble"],
                "fat_soluble": penalties["fat_soluble"]
            }
        },
        "food_quantities": {
            str(food): f"{float(amount):.1f}g"
            for food, amount in sorted(
                foods_consumed.items(),
                key=lambda x: float(x[1]),
                reverse=True
            )
        },
        "nutrition_profile": {
            str(nutrient): {
                "amount": f"{float(amount):.1f}{_get_unit(nutrient)}",
                "meal_percentage": f"{(float(amount)/float(meal_rdi[nutrient])*100):.1f}%",
                "daily_percentage": f"{(float(amount)/float(rdi[nutrient])*100):.1f}%",
                "status": "LOW" if float(amount)/float(meal_rdi[nutrient])*100 < 80
                         else "OK" if float(amount)/float(meal_rdi[nutrient])*100 < 150
                         else "HIGH"
            } for nutrient, amount in final_nutrients.items()
        },
        "summary": {
            "nutrients_at_good_levels": sum(1 for n, a in final_nutrients.items()
                                          if 80 <= float(a)/float(meal_rdi[n])*100 < 150),
            "nutrients_below_target": sum(1 for n, a in final_nutrients.items()
                                        if float(a)/float(meal_rdi[n])*100 < 80),
            "nutrients_above_target": sum(1 for n, a in final_nutrients.items()
                                        if float(a)/float(meal_rdi[n])*100 >= 150),
            "total_nutrients": len(meal_rdi)
        }
    }

    # Save JSON
    json_filename = f"recipes/json/meal_{run_number}_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Meal {run_number} - Nutrition Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .low {{ color: #dc3545; }}
        .ok {{ color: #28a745; }}
        .high {{ color: #ffc107; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .score {{ font-size: 1.2em; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Meal {run_number} Nutrition Report</h1>

    <div class="summary">
        <p><span class="score">Optimization Score: {score:.2f}</span></p>
        <p>Diet Type: {diet_type}</p>
        <p>Execution Time: {execution_time:.1f} seconds</p>
        <p>Generations: {generations}</p>
        <p>Foods Used: {len(foods_consumed)}</p>
    </div>

    <h2>Food Quantities</h2>
    <table>
        <tr><th>Food</th><th>Amount</th></tr>
        {''.join(f"<tr><td>{food}</td><td>{amount}</td></tr>"
                 for food, amount in report['food_quantities'].items())}
    </table>

    <h2>Nutrition Profile</h2>
    <table>
        <tr>
            <th>Nutrient</th>
            <th>Amount</th>
            <th>% of Meal Target</th>
            <th>% of Daily RDI</th>
            <th>Status</th>
        </tr>
        {''.join(f"<tr><td>{nutrient}</td>"
                 f"<td>{info['amount']}</td>"
                 f"<td>{info['meal_percentage']}</td>"
                 f"<td>{info['daily_percentage']}</td>"
                 f"<td class='{info['status'].lower()}'>{info['status']}</td></tr>"
                 for nutrient, info in report['nutrition_profile'].items())}
    </table>

    <h2>Summary</h2>
    <div class="summary">
        <p>Nutrients at good levels: {report['summary']['nutrients_at_good_levels']} of {report['summary']['total_nutrients']}</p>
        <p>Nutrients below target: {report['summary']['nutrients_below_target']} of {report['summary']['total_nutrients']}</p>
        <p>Nutrients above target: {report['summary']['nutrients_above_target']} of {report['summary']['total_nutrients']}</p>
    </div>
    <div class="penalties">
        <h2>Optimization Penalties</h2>
        <ul>
            <li>Under RDI: {penalties["under_rdi"]:.2f}x</li>
            <li>Over RDI: {penalties["over_rdi"]:.2f}x</li>
            <li>Over UL: {penalties["over_ul"]:.2f}x</li>
            <li>Water-soluble vitamins: {penalties["water_soluble"]:.2f}x</li>
            <li>Fat-soluble vitamins: {penalties["fat_soluble"]:.2f}x</li>
        </ul>
    </div>
</body>
</html>"""

    # Save HTML
    html_filename = f"recipes/html/meal_{run_number}_{timestamp}.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html)

    return json_filename

def print_nutrition_report(foods_consumed, food_data, rdi, number_of_meals=3, meal_number=1):
    """Print a detailed nutrition report based on foods consumed."""
    # Scale RDI for a single meal
    meal_rdi = {nutrient: target / number_of_meals for nutrient, target in rdi.items()}

    # Calculate nutrient totals
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}

    for food, amount in foods_consumed.items():
        for nutrient in meal_rdi:
            if nutrient in food_data[food]:
                nutrient_per_gram = food_data[food][nutrient] / food_data[food]['density']
                final_nutrients[nutrient] += nutrient_per_gram * amount

    # Print meal info
    print(f"\n=== MEAL {meal_number} OF {number_of_meals} ===")
    print(f"(Each meal targets {100/number_of_meals:.1f}% of daily nutrition needs)")

    # Print food quantities
    print("\nRecommended food quantities (in grams):")
    for food, amount in foods_consumed.items():
        print(f"{food}: {amount}g")

    # Print resulting nutrition profile
    print("\nNutrition profile for this meal:")
    print(f"{'Nutrient':<12} {'Amount':<8} {'% of Meal Target':<16} {'% of Daily RDI':<16} {'Status'}")
    print("-" * 70)

    for nutrient, amount in final_nutrients.items():
        meal_percentage = amount/meal_rdi[nutrient]*100
        daily_percentage = amount/rdi[nutrient]*100
        status = "LOW" if meal_percentage < 80 else "OK" if meal_percentage < 150 else "HIGH"
        print(f"{nutrient:<12} {amount:.1f}{_get_unit(nutrient):<3} {meal_percentage:.1f}%{' ':9} "
              f"{daily_percentage:.1f}%{' ':9} {status}")

    print("\nSummary:")
    nutrients_low = sum(1 for n, a in final_nutrients.items() if a/meal_rdi[n]*100 < 80)
    nutrients_ok = sum(1 for n, a in final_nutrients.items() if 80 <= a/meal_rdi[n]*100 < 150)
    nutrients_high = sum(1 for n, a in final_nutrients.items() if a/meal_rdi[n]*100 >= 150)

    print(f"- Nutrients at good levels: {nutrients_ok} of {len(meal_rdi)}")
    print(f"- Nutrients below target: {nutrients_low} of {len(meal_rdi)}")
    print(f"- Nutrients above target: {nutrients_high} of {len(meal_rdi)}")

def _get_unit(nutrient):
    """Get the appropriate unit for a nutrient."""
    # Look for unit in parentheses at end of nutrient name
    if '(mg)' in nutrient:
        return 'mg'
    elif '(ug)' in nutrient:
        return 'μg'
    elif '(g)' in nutrient:
        return 'g'
    else:
        return ''  # Default case

# Add this function before optimize_nutrition
def clean_column_name(col_name):
    """Clean column names by removing extra whitespace and newlines"""
    return col_name.strip().replace('\n', '')

def generate_index():
    """Generate both Markdown and HTML index files for meal plans"""
    print(f"Generating index files...")
    meals = []

    # Read all JSON files in the recipes directory
    recipes_dir = "recipes/json"
    for filename in os.listdir(recipes_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(recipes_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

                meal_info = data["meal_info"]
                summary = data["summary"]
                food_items = len(data["food_quantities"])

                timestamp_str = '_'.join(filename.split('_')[2:]).split('.')[0]
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                diet_type = meal_info.get("diet_type", "all")

                # Handle missing penalties field with default values
                default_penalties = {
                    "under_rdi": 1.5,
                    "over_rdi": 0.5,
                    "over_ul": 2.5,
                    "water_soluble": 0.5,
                    "fat_soluble": 0.8
                }

                meals.append({
                    "filename": filename,
                    "run_number": meal_info["run_number"],
                    "diet_type": diet_type,
                    "optimization_score": meal_info["optimization_score"],
                    "nutrients_ok": summary["nutrients_at_good_levels"],
                    "nutrients_low": summary["nutrients_below_target"],
                    "nutrients_high": summary["nutrients_above_target"],
                    "food_items": food_items,
                    "timestamp": timestamp,
                    "generations": meal_info["generations"],
                    "execution_time": meal_info["execution_time_seconds"],
                    'penalties': meal_info.get('penalties', default_penalties)
                })

    # Sort meals by optimization score (ascending - better scores first)
    meals.sort(key=lambda x: x["optimization_score"])

    # Generate Markdown
    markdown = "# Genetic Algorithm Optimised Nutrition Recipes\n\n"
    markdown += "Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
    markdown += "| Run # | Diet | Score | Foods | Nutrients (OK/Low/High) | Generations | Time (s) | Filename |\n"
    markdown += "|-------|------|-------|-------|----------------------|------------|----------|----------|\n"

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Genetic Algorithm Optimised Nutrition Recipes</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <a href="https://github.com/alexlaverty/optimize-nutrition" target="_blank"><img src="forkme_right_darkblue_121621.svg" style="position:absolute;top:0;right:0;" alt="Fork me on GitHub"></a>
    <h1>Genetic Algorithm Optimised Nutrition Recipes</h1>
    <div class="description">
        <p>This table shows meal plans optimized using a genetic algorithm to meet daily nutritional requirements.
        Each row represents a different optimization run, where lower scores indicate better nutritional balance.
        The "Diet" column shows the type of diet (all foods, vegan, or whole food plant-based),
        "Foods" shows how many ingredients were used, and "Nutrients" shows how many nutrients are at good levels,
        below target, and above target respectively. Click on any row to view the detailed recipe and nutritional analysis.</p>
    </div>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <table>
        <tr>
            <th>#</th>
            <th>Diet</th>
            <th>Score</th>
            <th>Foods</th>
            <th>Nutrients (OK/Low/High)</th>
            <th>Generations</th>
            <th>Time (s)</th>
            <th>View Recipe</th>
        </tr>
"""

    for idx, meal in enumerate(meals, start=1):
        # Add markdown row
        markdown += f"| {meal['run_number']} | {meal['diet_type']} | {meal['optimization_score']:.2f} | "
        markdown += f"{meal['food_items']} | {meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']} | "
        markdown += f"{meal['generations']} | {meal['execution_time']:.1f} | "
        markdown += f"[{os.path.basename(meal['filename'])}](recipes/html/{os.path.splitext(meal['filename'])[0]}.html) |\n"

        # Add HTML row
        score_class = 'good' if meal['optimization_score'] < 5 else 'warning' if meal['optimization_score'] < 10 else 'error'
        html += f"""        <tr>
            <td>{idx}</td>
            <td>{meal['diet_type']}</td>
            <td class="{score_class}">{meal['optimization_score']:.2f}</td>
            <td>{meal['food_items']}</td>
            <td>{meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']}</td>
            <td>{meal['generations']}</td>
            <td>{meal['execution_time']:.1f}</td>
            <td><a href="{os.path.splitext(meal['filename'])[0]}.html">{os.path.basename(meal['filename'])}</a></td>
        </tr>
"""

    # Close HTML
    html += """    </table>
</body>
</html>
"""

    # Save both files
    with open("README.md", "w") as f:
        f.write(markdown)

    with open("recipes/html/index.html", "w") as f:
        f.write(html)

    print(f"Generated README.md and recipes/html/index.html")

def init_directories():
    """Initialize directory structure for recipes and output files."""
    base_dir = 'recipes'
    subdirs = ['json', 'html']

    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def cleanup_high_score_recipes(max_score=20, max_files=250):
    """
    Remove recipe files based on score threshold and total file count limit.

    Args:
        max_score (float): Maximum allowed optimization score (default: 20)
        max_files (int): Maximum number of recipe files to keep (default: 250)
    """
    print(f"\nCleaning up recipes (max score: {max_score}, max files: {max_files})...")
    recipes = []
    removed_count = 0

    # Collect all recipe data
    recipes_dir = "recipes/json"
    for filename in os.listdir(recipes_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(recipes_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    score = data["meal_info"]["optimization_score"]
                    recipes.append({
                        "filename": filename,
                        "score": score,
                        "filepath": filepath
                    })
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Sort recipes by score (best scores first)
    recipes.sort(key=lambda x: x["score"])

    # Process recipes
    for idx, recipe in enumerate(recipes):
        should_remove = False
        reason = ""

        if recipe["score"] > max_score:
            should_remove = True
            reason = f"score too high ({recipe['score']:.2f} > {max_score})"
        elif idx >= max_files:
            should_remove = True
            reason = f"exceeds file limit ({idx + 1} > {max_files})"

        if should_remove:
            # Remove JSON file
            os.remove(recipe["filepath"])

            # Remove corresponding HTML file
            html_file = os.path.join("recipes/html",
                                   os.path.splitext(recipe["filename"])[0] + ".html")
            if os.path.exists(html_file):
                os.remove(html_file)

            removed_count += 1
            print(f"Removed recipe {recipe['filename']} ({reason})")

    remaining_count = len(recipes) - removed_count
    print(f"\nCleanup complete:")
    print(f"- Removed {removed_count} recipes")
    print(f"- {remaining_count} recipes remaining")
    print(f"- Best score: {recipes[0]['score']:.2f}")
    print(f"- Worst remaining score: {recipes[min(remaining_count-1, len(recipes)-1)]['score']:.2f}")


if __name__ == "__main__":
    start_time = time.time()
    init_directories()
    # Load the Excel file
    file_path = "Release 2 - Nutrient file.xlsx"
    df = pd.read_excel(file_path, sheet_name="All solids & liquids per 100g")

    # Load the RDI data
    with open('rdi.json', 'r') as f:
        nutrient_mapping = json.load(f)

    # Settings that will be the same for all runs
    number_of_meals = 1
    meal_number = 1
    rdi_values = {nutrient: details['rdi'] for nutrient, details in nutrient_mapping.items()}

    # Run for each diet type
    for run_type in ['all', 'vegan', 'wfpb', 'nutrient_dense']:
    #for run_type in ['nutrient_dense']:
        print(f"\n=== Starting {run_type.upper()} foods optimization ===")

        # Filter foods based on diet type
        working_df = df.copy()
        if run_type != 'all':
            try:
                # Load diet-specific rules
                config_file = f'{run_type}.json'
                with open(config_file, 'r') as f:
                    diet_config = json.load(f)

                initial_food_count = len(working_df)

                # First apply inclusion filter if specified
                if 'included_terms' in diet_config:
                    # Convert all included terms to lowercase once
                    included_terms = [term.lower() for term in diet_config['included_terms']]

                    # Create inclusion mask using lowercase comparison
                    inclusion_mask = working_df['Food Name'].str.lower().apply(
                        lambda x: any(term in x for term in included_terms)
                    )
                    working_df = working_df[inclusion_mask]
                    print(f"Included {len(working_df)} {run_type} foods based on inclusion criteria")

                # Then apply exclusion filter
                excluded_terms = diet_config.get('excluded_terms', [])
                if excluded_terms:
                    # Create exclusion mask - remove foods that match any excluded term
                    exclusion_mask = ~working_df['Food Name'].str.lower().apply(
                        lambda x: any(term.lower() in x for term in excluded_terms)
                    )
                    working_df = working_df[exclusion_mask]
                    print(f"Removed {initial_food_count - len(working_df)} foods based on exclusion criteria")

                print(f"Final food count: {len(working_df)} foods")

                if len(working_df) == 0:
                    print(f"Warning: No foods remain after filtering for {run_type} diet")
                    continue

            except FileNotFoundError:
                print(f"Warning: {config_file} not found, skipping {run_type} optimization")
                continue

        # Randomly select between 5-20 foods
        n_foods = random.randint(10, 30)
        random_foods = working_df.sample(n=n_foods)
        print(f"Selected {n_foods} random foods for optimization")

        # Generate random number of generations
        generations = random.randint(10, 300)
        generations = 3
        print(f"Selected {generations} for number of generations")

        # Clean column names
        random_foods.columns = [clean_column_name(col) for col in random_foods.columns]

        # Run optimization
        result = optimize_nutrition(
            food_df=random_foods,
            nutrient_mapping=nutrient_mapping,
            rdi_targets=rdi_values,
            number_of_meals=number_of_meals,
            meal_number=meal_number,
            randomness_factor=0.4,
            population_size=100,
            generations=generations,
            diet_type=run_type
        )

        print(f"=== Completed {run_type.upper()} foods optimization ===\n")

    cleanup_high_score_recipes(max_score=20, max_files=250)
    # Generate final index
    generate_index()