import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd

def generate_optimization_insights():
    """Generate visualization charts for optimization analysis."""
    recipes = []
    recipes_dir = "recipes/json"

    # Collect data from all recipe files
    for filename in os.listdir(recipes_dir):
        if filename.endswith(".json"):
            with open(os.path.join(recipes_dir, filename), 'r') as f:
                data = json.load(f)
                recipes.append({
                    'score': data['meal_info']['optimization_score'],
                    'food_count': len(data['food_quantities']),
                    'generations': data['meal_info']['generations'],
                    'diet_type': data['meal_info']['diet_type'],
                    'execution_time': data['meal_info']['execution_time_seconds'],
                    'nutrients_ok': data['summary']['nutrients_at_good_levels'],
              })

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(recipes)

    # Create subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Score vs Food Count scatter plot
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=df, x='food_count', y='score', hue='diet_type', alpha=0.6)
    plt.title('Optimization Score vs Number of Foods')
    plt.xlabel('Number of Foods')
    plt.ylabel('Score (lower is better)')

    # 2. Score vs Generations scatter plot
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df, x='generations', y='score', hue='diet_type', alpha=0.6)
    plt.title('Score vs Number of Generations')
    plt.xlabel('Generations')
    plt.ylabel('Score')

    # 3. Score distribution by diet type
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='diet_type', y='score')
    plt.title('Score Distribution by Diet Type')
    plt.xticks(rotation=45)

    # 4. Correlation between penalties and score
    # plt.subplot(2, 3, 4)
    # penalties_df = pd.DataFrame([r['penalties'] for r in df['penalties']])
    # correlations = pd.concat([df['score'], penalties_df], axis=1).corr()['score'].drop('score')
    # sns.barplot(x=correlations.index, y=correlations.values)
    # plt.title('Correlation of Penalties with Score')
    # plt.xticks(rotation=45)

    # 5. Score vs Nutrients at good levels
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df, x='nutrients_ok', y='score', hue='diet_type', alpha=0.6)
    plt.title('Score vs Nutrients at Good Levels')
    plt.xlabel('Number of Nutrients at Good Levels')
    plt.ylabel('Score')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('optimization_insights.png')
    print("Generated optimization insights chart")

    # Generate summary statistics
    summary = {
        'best_scores': df.nsmallest(5, 'score')[['score', 'food_count', 'generations', 'diet_type']],
        'optimal_ranges': {
            'food_count': {
                'mean': df[df['score'] < df['score'].median()]['food_count'].mean(),
                'std': df[df['score'] < df['score'].median()]['food_count'].std()
            },
            'generations': {
                'mean': df[df['score'] < df['score'].median()]['generations'].mean(),
                'std': df[df['score'] < df['score'].median()]['generations'].std()
            }
        }
    }

    # # Save summary as JSON
    # with open('recipes/html/optimization_summary.json', 'w') as f:
    #     json.dump(summary, f, indent=2)


if __name__ == "__main__":
    generate_optimization_insights()  # Add this line
