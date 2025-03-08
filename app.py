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

def optimize_nutrition(food_df, nutrient_mapping, rdi_targets,
                       number_of_meals=1, meal_number=1,
                       randomness_factor=0.3, population_size=50,
                       generations=100):
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

    def evaluate_solution(solution):
        """Evaluate how close the solution is to the RDI targets."""
        current_nutrients = {nutrient: 0 for nutrient in meal_rdi_targets}
        for food, amount in solution.items():
            for nutrient in meal_rdi_targets:
                if nutrient in foods_df.loc[food]:
                    nutrient_per_gram = foods_df.loc[food][nutrient] / foods_df.loc[food]['density']
                    current_nutrients[nutrient] += nutrient_per_gram * amount
        return _calculate_nutrition_score(current_nutrients, meal_rdi_targets)

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

    execution_time = time.time() - start_time  # Calculate total time
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
                                        execution_time=execution_time)

    print(f"Generation {generation + 1}/{generations}, Best Score: {best_score}")
    print(f"Report saved to: {report_file}")

    # Round the amounts to make them more practical
    return {food: round(amount) for food, amount in best_solution.items() if amount > 10}

def _is_nutrition_sufficient(current, targets, threshold=0.90):
    """Check if current nutrients are at least threshold percent of targets."""
    for nutrient, target in targets.items():
        if current.get(nutrient, 0) < threshold * target:
            return False
    return True

def _calculate_nutrition_score(current, targets):
    """Calculate how far current nutrients are from targets."""
    score = 0
    for nutrient, target in targets.items():
        # Penalize both under and over consumption, but under more heavily
        if current.get(nutrient, 0) < target:
            score += ((target - current.get(nutrient, 0)) / target) ** 2 * 1.5
        else:
            # For vitamins and minerals, we're usually more tolerant of excess
            if nutrient in ['vitamin_a', 'vitamin_c', 'vitamin_e', 'vitamin_k', 'vitamin_b1', 'vitamin_b2',
                           'vitamin_b3', 'vitamin_b5', 'vitamin_b6', 'vitamin_b12', 'folate']:
                # Be more lenient for water-soluble vitamins (less penalty for excess)
                excess_factor = 0.5 if nutrient in ['vitamin_c', 'vitamin_b1', 'vitamin_b2',
                                                   'vitamin_b3', 'vitamin_b5', 'vitamin_b6',
                                                   'vitamin_b12', 'folate'] else 0.8
                score += ((current.get(nutrient, 0) - target) / target) ** 2 * excess_factor
            else:
                score += ((current.get(nutrient, 0) - target) / target) ** 2
    return score

def save_nutrition_report(foods_consumed, food_data, rdi, score, run_number,
                          number_of_meals=3, meal_number=1, generations=100,
                          execution_time=0):
    """Save a detailed nutrition report as JSON."""
    # Create recipes directory if it doesn't exist
    if not os.path.exists('recipes'):
        os.makedirs('recipes')

    # Scale RDI for a single meal
    meal_rdi = {nutrient: float(target) / number_of_meals for nutrient, target in rdi.items()}

    # Calculate nutrient totals
    final_nutrients = {nutrient: 0 for nutrient in meal_rdi}

    for food, amount in foods_consumed.items():
        for nutrient in meal_rdi:
            if nutrient in food_data[food]:
                nutrient_per_gram = float(food_data[food][nutrient]) / float(food_data[food]['density'])
                final_nutrients[nutrient] += nutrient_per_gram * amount

    # Create report dictionary with explicit float conversions
    report = {
        "meal_info": {
            "run_number": int(run_number),
            "target_percentage": float(100/number_of_meals),
            "optimization_score": float(score),
            "generations": int(generations),
            "number_of_foods": len(foods_consumed),
            "execution_time_seconds": float(execution_time)
        },
        "food_quantities": {
            str(food): f"{float(amount):.1f}g" for food, amount in foods_consumed.items()
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

    filename = f"recipes/meal_{run_number}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    return filename

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
    if nutrient == 'protein' or nutrient == 'fiber':
        return 'g'
    elif nutrient in ['vitamin_a', 'vitamin_k', 'folate']:
        return 'μg'
    else:
        return 'mg'

# Add this function before optimize_nutrition
def clean_column_name(col_name):
    """Clean column names by removing extra whitespace and newlines"""
    return col_name.strip().replace('\n', '')

def generate_index():
    print(f"Generating README.md")
    # Create a list to store meal data
    meals = []

    # Read all JSON files in the recipes directory
    recipes_dir = "recipes"
    for filename in os.listdir(recipes_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(recipes_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

                # Extract relevant information
                meal_info = data["meal_info"]
                summary = data["summary"]
                food_items = len(data["food_quantities"])

                # Parse timestamp from filename (format: meal_15_20250308_215231.json)
                timestamp_str = '_'.join(filename.split('_')[2:]).split('.')[0]  # Gets "20250308_215231"
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                #print(meal_info)
                meals.append({
                    "filename": filename,
                    "run_number": meal_info["run_number"],
                    "optimization_score": meal_info["optimization_score"],
                    "nutrients_ok": summary["nutrients_at_good_levels"],
                    "nutrients_low": summary["nutrients_below_target"],
                    "nutrients_high": summary["nutrients_above_target"],
                    "food_items": food_items,
                    "timestamp": timestamp,
                    "generations": meal_info["generations"],
                    "execution_time": meal_info["execution_time_seconds"]
                })

    # Sort meals by optimization score (ascending - better scores first)
    meals.sort(key=lambda x: x["optimization_score"])

    markdown = "# Meal Plan Index\n\n"
    markdown += "Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
    markdown += "| Run # | Score | Foods | Nutrients (OK/Low/High) | Generations | Time (s) | Filename |\n"
    markdown += "|-------|-------|-------|----------------------|------------|----------|----------|\n"

    for meal in meals:
        markdown += f"| {meal['run_number']} | {meal['optimization_score']:.2f} | "
        markdown += f"{meal['food_items']} | {meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']} | "
        markdown += f"{meal['generations']} | {meal['execution_time']:.1f} | "
        markdown += f"[{os.path.basename(meal['filename'])}](recipes/{meal['filename']}) |\n"

    # Save the index file
    with open("README.md", "w") as f:
        f.write(markdown)



if __name__ == "__main__":
    start_time = time.time()
    # Load the Excel file
    file_path = "Release 2 - Nutrient file.xlsx"
    df = pd.read_excel(file_path, sheet_name="All solids & liquids per 100g")

    # Randomly select between 5-20 foods
    n_foods = random.randint(5, 20)
    random_foods = df.sample(n=n_foods)
    print(f"Selected {n_foods} random foods for optimization")

    generations = random.randint(100, 300)
    #generations = 3
    print(f"Selected {generations} for number of generations")

    # Clean column names
    random_foods.columns = [clean_column_name(col) for
                            col in random_foods.columns]

    # Load the RDI data
    with open('rdi.json', 'r') as f:
        nutrient_mapping = json.load(f)

    # Number of meals per day
    number_of_meals = 1

    # Run the optimizer for a specific meal (e.g., meal 1 of 1)
    meal_number = 1
    rdi_values = {nutrient: details['rdi'] for nutrient,
                  details in nutrient_mapping.items()}

    result = optimize_nutrition(food_df=random_foods,
                              nutrient_mapping=nutrient_mapping,
                              rdi_targets=rdi_values,
                              number_of_meals=number_of_meals,
                              meal_number=meal_number,
                              randomness_factor=0.4,
                              population_size=100,
                              generations=generations)

    generate_index()