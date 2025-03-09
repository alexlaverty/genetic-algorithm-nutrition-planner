import pandas as pd
import json

def print_random_foods(diet_type='all', count=25):
    """
    Print random foods from the nutrient database, filtered by diet type.

    Args:
        diet_type: str - 'all', 'vegan', or 'wfpb'
        count: int - number of random foods to display
    """
    # Load the Excel file
    file_path = "Release 2 - Nutrient file.xlsx"
    df = pd.read_excel(file_path, sheet_name="All solids & liquids per 100g")

    # Filter foods based on diet type
    if diet_type != 'all':
        try:
            # Load diet-specific exclusion rules
            config_file = f'{diet_type}.json'
            with open(config_file, 'r') as f:
                exclusion_config = json.load(f)
                excluded_terms = exclusion_config['excluded_terms']

            # Create a filter mask for non-excluded foods
            mask = ~df['Food Name'].str.lower().apply(
                lambda x: any(term.lower() in x for term in excluded_terms)
            )
            df = df[mask]
            print(f"\nFiltered out non-{diet_type} foods. {len(df)} foods remaining.")
        except FileNotFoundError:
            print(f"Warning: {config_file} not found, showing all foods")

    # Get random sample
    count = min(count, len(df))  # Ensure we don't try to sample more foods than available
    random_foods = df.sample(n=count)

    # Print foods with index numbers
    print(f"\n{count} Random {diet_type.upper()} Foods:")
    print("-" * 50)
    for i, food in enumerate(random_foods['Food Name'], 1):
        print(f"{i:2d}. {food}")

if __name__ == "__main__":
    # You can change this to 'all', 'vegan', or 'wfpb'
    print_random_foods(diet_type='wfpb', count=25)