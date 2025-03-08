import json
import os
from datetime import datetime

def generate_index():
    # Create a list to store meal data
    meals = []

    # Read all JSON files in the results directory
    results_dir = "results"
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

                # Extract relevant information
                meal_info = data["meal_info"]
                summary = data["summary"]
                food_items = len(data["food_quantities"])

                # Parse timestamp from filename (format: meal_15_20250308_215231.json)
                timestamp_str = '_'.join(filename.split('_')[2:]).split('.')[0]  # Gets "20250308_215231"
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

                meals.append({
                    "filename": filename,
                    "run_number": meal_info["run_number"],
                    "optimization_score": meal_info["optimization_score"],
                    "nutrients_ok": summary["nutrients_at_good_levels"],
                    "nutrients_low": summary["nutrients_below_target"],
                    "nutrients_high": summary["nutrients_above_target"],
                    "food_items": food_items,
                    "timestamp": timestamp
                })

    # Sort meals by optimization score (ascending - better scores first)
    meals.sort(key=lambda x: x["optimization_score"])

    # Generate markdown table
    markdown = "# Meal Plan Index\n\n"
    markdown += "Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
    markdown += "| Run # | Score | Foods | Nutrients (OK/Low/High) | Date | Filename |\n"
    markdown += "|-------|-------|-------|----------------------|------|----------|\n"

    for meal in meals:
        markdown += f"| {meal['run_number']} | {meal['optimization_score']:.2f} | "
        markdown += f"{meal['food_items']} | {meal['nutrients_ok']}/{meal['nutrients_low']}/{meal['nutrients_high']} | "
        markdown += f"{meal['timestamp'].strftime('%Y-%m-%d %H:%M')} | [{os.path.basename(meal['filename'])}]({meal['filename']}) |\n"

    # Save the index file
    with open("README.md", "w") as f:
        f.write(markdown)

    print(f"Index generated at README.md")

if __name__ == "__main__":
    generate_index()