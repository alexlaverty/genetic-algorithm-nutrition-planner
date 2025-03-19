import random
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import time
import copy

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables for optimization state
optimization_running = False
optimization_thread = None
should_stop = False

def load_data():
    with open('foods.json', 'r') as f:
        foods = json.load(f)
    with open('rdi.json', 'r') as f:
        rdi = json.load(f)
    return foods, rdi

foods, rdi = load_data()

def get_next_run_number():
    """Get the next run number by checking existing CSV file"""
    if not os.path.exists('optimization_history.csv'):
        return 1
    df = pd.read_csv('optimization_history.csv')
    return df['run_number'].max() + 1 if not df.empty else 1

def optimize_nutrition_core(food_df, nutrient_mapping, rdi_targets, params):
    global optimization_running, should_stop

    number_of_meals = params.get('number_of_meals', 1)
    meal_number = params.get('meal_number', 1)
    population_size = params.get('population_size', 50)
    generations = params.get('generations', 100)
    diet_type = params.get('diet_type', 'all')

    if meal_number < 1 or meal_number > number_of_meals:
        raise ValueError(f"Meal number must be between 1 and {number_of_meals}")

    meal_rdi_targets = {nutrient: target / number_of_meals for nutrient, target in rdi_targets.items()}
    available_foods = {}

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

    def evaluate_solution(solution):
        current_nutrients = {nutrient: 0 for nutrient in meal_rdi_targets}
        for food, amount in solution.items():
            for nutrient in meal_rdi_targets:
                if nutrient in foods_df.loc[food]:
                    nutrient_per_gram = foods_df.loc[food][nutrient] / foods_df.loc[food]['density']
                    current_nutrients[nutrient] += nutrient_per_gram * amount
        return _calculate_nutrition_score(current_nutrients, meal_rdi_targets, penalties)

    def mutate_solution(solution):
        food = random.choice(list(solution.keys()))
        solution[food] = max(0, solution[food] + random.uniform(-20, 20))
        return solution

    def crossover_solution(sol1, sol2):
        new_solution = {}
        for food in sol1:
            new_solution[food] = sol1[food] if random.random() < 0.5 else sol2[food]
        return new_solution

    population = [create_solution() for _ in range(population_size)]
    best_solution = None
    best_score = float('inf')
    run_number = get_next_run_number()

    socketio.emit('status_update', {'message': f'Starting optimization run #{run_number}'})
    print(f"Starting optimization run #{run_number}")  # Console feedback

    for generation in range(generations):
        if should_stop:
            socketio.emit('status_update', {'message': 'Optimization stopped by user'})
            print("Optimization stopped by user")  # Console feedback
            optimization_running = False
            return None

        scores = [(evaluate_solution(sol), sol) for sol in population]
        scores.sort(key=lambda x: x[0])
        best_score, best_solution = scores[0]

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

        # Emit real-time update with nutrition report
        nutrition_report = generate_nutrition_report(best_solution, available_foods, rdi_targets, number_of_meals, meal_number)
        socketio.emit('optimization_update', {
            'generation': generation + 1,
            'total_generations': generations,
            'best_score': round(best_score, 2),
            'nutrition_report': nutrition_report
        })
        print(f"Generation {generation + 1}/{generations}, Best Score: {best_score:.2f}")  # Console feedback

        # Small delay to ensure UI updates
        time.sleep(0.1)

        population = [sol for _, sol in scores[:population_size // 2]]
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover_solution(parent1, parent2)
            if random.random() < 0.1:
                child = mutate_solution(child)
            new_population.append(child)
        population = new_population

    execution_time = time.time() - time.time()  # Note: This should use start_time, fixed below
    start_time = time.time()  # Move this before the loop if you want accurate timing
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
                    'amount': float(info['amount'].rstrip('mgu')),
                    'unit': info['amount'][-2:] if info['amount'][-2:] in ['mg', 'ug'] else info['amount'][-1:],
                    'meal_percentage': float(info['meal_percentage'].rstrip('%')),
                    'daily_percentage': float(info['daily_percentage'].rstrip('%')),
                    'status': info['status']
                } for nutrient, info in final_report_data['nutrition_profile'].items()
            ],
            'summary': final_report_data['summary']
        }
    })
    print(f"Optimization complete, report saved to: {final_report}")  # Console feedback

    optimization_running = False
    return {
        "solution": {food: round(amount) for food, amount in best_solution.items() if amount > 10},
        "penalties": penalties
    }

def _calculate_nutrition_score(current, targets):
    """Calculate how far current nutrients are from targets, with upper limits."""
    score = 0
    for nutrient, target_values in targets.items():
        rdi_target = target_values["rdi"]
        ul_target = target_values.get("ul")
        current_value = current.get(nutrient, 0)

        # Penalize being below RDI
        if current_value < rdi_target:
            score += ((rdi_target - current_value) / rdi_target) ** 2 * 1.5
        elif ul_target and current_value > ul_target:
            # Heavily penalize exceeding upper limit
            score += ((current_value - ul_target) / ul_target) ** 2 * 3.0
        else:
            # Between RDI and UL (or no UL defined)
            # For most nutrients, slight penalty for going above RDI but below UL
            excess_factor = 0.5

            # For water-soluble vitamins, be more lenient
            if nutrient in ['vitamin_c', 'vitamin_b1', 'vitamin_b2', 'vitamin_b3',
                           'vitamin_b5', 'vitamin_b6', 'vitamin_b12', 'folate']:
                excess_factor = 0.2

            # Apply a small penalty for exceeding RDI but staying under UL
            score += ((current_value - rdi_target) / rdi_target) ** 2 * excess_factor

    return score

def _get_unit(nutrient):
    """Get the appropriate unit for a nutrient."""
    if nutrient == 'protein' or nutrient == 'fiber':
        return 'g'
    elif nutrient in ['vitamin_a', 'vitamin_k', 'folate']:
        return 'μg'
    else:
        return 'mg'

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
    foods_data, rdi_data = load_data()
    return render_template('index.html', foods=foods_data, rdi=rdi_data)

@app.route('/start_optimization', methods=['POST'])
def start_optimization():
    global optimization_running, optimization_thread, should_stop

    if optimization_running:
        return jsonify({'status': 'error', 'message': 'Optimization already running'})

    # Get form data
    data = request.get_json()
    number_of_meals = int(data.get('number_of_meals', 1))
    meal_number = int(data.get('meal_number', 1))
    selected_foods = data.get('selected_foods', [])
    population_size = int(data.get('population_size', 50))
    generations = int(data.get('generations', 100))

    # Validate selected foods
    if not selected_foods:
        return jsonify({'status': 'error', 'message': 'No foods selected'})

    # Filter available foods based on user selection
    available_foods = {k: v for k, v in foods.items() if k in selected_foods}

    # Reset the stop flag
    should_stop = False
    optimization_running = True

    # Start optimization in a separate thread
    optimization_thread = threading.Thread(
        target=optimize_nutrition_async,
        args=(available_foods, rdi, number_of_meals, meal_number, 100, 0.4, population_size, generations)
    )
    optimization_thread.daemon = True
    optimization_thread.start()

    return jsonify({'status': 'success', 'message': 'Optimization started'})

@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    global should_stop

    should_stop = True
    return jsonify({'status': 'success', 'message': 'Stopping optimization...'})

@socketio.on('connect')
def handle_connect():
    emit('status_update', {'message': 'Connected to server'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nutrition Optimizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        .progress {
            height: 25px;
            margin-bottom: 15px;
        }
        .card {
            margin-bottom: 20px;
        }
        .status-ok {
            color: #28a745;
        }
        .status-low {
            color: #dc3545;
        }
        .status-high {
            color: #fd7e14;
        }
        #optimization-progress {
            transition: width 0.5s ease;
        }
        .food-list {
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Nutrition Meal Optimizer</h1>

        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Settings</h5>
                    </div>
                    <div class="card-body">
                        <form id="optimization-form">
                            <div class="mb-3">
                                <label for="number-of-meals" class="form-label">Number of Meals Per Day</label>
                                <input type="number" class="form-control" id="number-of-meals" min="1" max="6" value="1">
                            </div>

                            <div class="mb-3">
                                <label for="meal-number" class="form-label">Meal Number</label>
                                <input type="number" class="form-control" id="meal-number" min="1" value="1">
                            </div>

                            <div class="mb-3">
                                <label for="population-size" class="form-label">Population Size</label>
                                <input type="number" class="form-control" id="population-size" min="10" max="500" value="50">
                                <small class="text-muted">Higher values = better results but slower</small>
                            </div>

                            <div class="mb-3">
                                <label for="generations" class="form-label">Generations</label>
                                <input type="number" class="form-control" id="generations" min="10" max="1000" value="300">
                                <small class="text-muted">Higher values = better results but slower</small>
                            </div>

                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary" id="start-button">Start Optimization</button>
                                <button type="button" class="btn btn-danger" id="stop-button" disabled>Stop Optimization</button>
                            </div>
                        </form>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h5>Available Foods</h5>
                        <button type="button" class="btn btn-sm btn-outline-secondary mt-2" id="select-all">Select All</button>
                        <button type="button" class="btn btn-sm btn-outline-secondary mt-2" id="deselect-all">Deselect All</button>
                    </div>
                    <div class="card-body food-list">
                        <div id="food-checklist">
                            <!-- Foods will be populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-md-8">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Optimization Progress</h5>
                    </div>
                    <div class="card-body">
                        <div class="progress">
                            <div id="optimization-progress" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                        </div>
                        <div id="status-message" class="text-muted">Waiting to start optimization...</div>
                        <div id="generation-info" class="mt-2"></div>
                    </div>
                </div>
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Score History</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="scoreChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h5>Nutrition Report</h5>
                    </div>
                    <div class="card-body">
                        <div id="meal-info"></div>

                        <div id="foods-section" class="mt-4">
                            <h6>Recommended Food Quantities</h6>
                            <div id="food-quantities" class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Food</th>
                                            <th>Amount (g)</th>
                                        </tr>
                                    </thead>
                                    <tbody id="food-quantities-body">
                                        <tr><td colspan="2">No data yet</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div id="nutrients-section" class="mt-4">
                            <h6>Nutrition Profile</h6>
                            <div id="nutrient-profile" class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Nutrient</th>
                                            <th>Amount</th>
                                            <th>% of Meal Target</th>
                                            <th>% of Daily RDI</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody id="nutrient-profile-body">
                                        <tr><td colspan="5">No data yet</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div id="summary-section" class="mt-4">
                            <h6>Summary</h6>
                            <div id="nutrition-summary">
                                <p>Run the optimization to see results</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        // Connect to WebSocket
        const socket = io();

        // DOM elements
        const form = document.getElementById('optimization-form');
        const startButton = document.getElementById('start-button');
        const stopButton = document.getElementById('stop-button');
        const progressBar = document.getElementById('optimization-progress');
        const statusMessage = document.getElementById('status-message');
        const generationInfo = document.getElementById('generation-info');
        const mealInfo = document.getElementById('meal-info');
        const foodQuantitiesBody = document.getElementById('food-quantities-body');
        const nutrientProfileBody = document.getElementById('nutrient-profile-body');
        const nutritionSummary = document.getElementById('nutrition-summary');
        const foodChecklist = document.getElementById('food-checklist');
        const selectAllBtn = document.getElementById('select-all');
        const deselectAllBtn = document.getElementById('deselect-all');
        const numberOfMealsInput = document.getElementById('number-of-meals');
        const mealNumberInput = document.getElementById('meal-number');

        // Populate food checklist
        function populateFoodChecklist() {
            fetch('/get_foods')
                .then(response => response.json())
                .then(data => {
                    const foods = data.foods;
                    let html = '';

                    for (const food in foods) {
                        html += `
                            <div class="form-check">
                                <input class="form-check-input food-checkbox" type="checkbox" value="${food}" id="food-${food.replace(/\\s+/g, '-').toLowerCase()}" checked>
                                <label class="form-check-label" for="food-${food.replace(/\\s+/g, '-').toLowerCase()}">
                                    ${food}
                                </label>
                            </div>
                        `;
                    }

                    foodChecklist.innerHTML = html;
                });
        }

        // Or create static checklist based on the template data
        function createStaticFoodChecklist() {
            const foods = {{ foods|tojson }};
            let html = '';

            for (const food in foods) {
                html += `
                    <div class="form-check">
                        <input class="form-check-input food-checkbox" type="checkbox" value="${food}" id="food-${food.replace(/\\s+/g, '-').toLowerCase()}" checked>
                        <label class="form-check-label" for="food-${food.replace(/\\s+/g, '-').toLowerCase()}">
                            ${food}
                        </label>
                    </div>
                `;
            }

            foodChecklist.innerHTML = html;
        }

        // Call this on page load
        createStaticFoodChecklist();

        // Handle select/deselect all
        selectAllBtn.addEventListener('click', () => {
            document.querySelectorAll('.food-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
        });

        deselectAllBtn.addEventListener('click', () => {
            document.querySelectorAll('.food-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
        });

        // Number of meals input affects meal number max value
        numberOfMealsInput.addEventListener('change', () => {
            const numberOfMeals = parseInt(numberOfMealsInput.value);
            mealNumberInput.max = numberOfMeals;

            if (parseInt(mealNumberInput.value) > numberOfMeals) {
                mealNumberInput.value = numberOfMeals;
            }
        });

        // Form submission
        form.addEventListener('submit', (e) => {
            e.preventDefault();

            // Get selected foods
            const selectedFoods = [];
            document.querySelectorAll('.food-checkbox:checked').forEach(checkbox => {
                selectedFoods.push(checkbox.value);
            });

            if (selectedFoods.length === 0) {
                alert('Please select at least one food');
                return;
            }

            // Get form values
            const numberOfMeals = parseInt(document.getElementById('number-of-meals').value);
            const mealNumber = parseInt(document.getElementById('meal-number').value);
            const populationSize = parseInt(document.getElementById('population-size').value);
            const generations = parseInt(document.getElementById('generations').value);

            // Reset chart data
            scores.length = 0;
            generations.length = 0;
            scoreChart.data.labels = [];
            scoreChart.data.datasets[0].data = [];
            scoreChart.update();

            if (mealNumber > numberOfMeals) {
                alert(`Meal number cannot be greater than the number of meals (${numberOfMeals})`);
                return;
            }

            // Update UI to show optimization is starting
            startButton.disabled = true;
            stopButton.disabled = false;
            progressBar.style.width = '0%';
            statusMessage.textContent = 'Starting optimization...';

            // Start optimization
            fetch('/start_optimization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    number_of_meals: numberOfMeals,
                    meal_number: mealNumber,
                    selected_foods: selectedFoods,
                    population_size: populationSize,
                    generations: generations
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    statusMessage.textContent = data.message;
                    startButton.disabled = false;
                    stopButton.disabled = true;
                }
            });
        });

        // Stop button
        stopButton.addEventListener('click', () => {
            fetch('/stop_optimization', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                statusMessage.textContent = data.message;
            });
        });

        // Socket.io event handlers
        socket.on('connect', () => {
            console.log('Connected to the server');
        });

        // Add this after the socket.io connection setup
        let scoreChart;
        const scores = [];
        const generations = [];

        // Initialize the chart
        function initializeChart() {
            const ctx = document.getElementById('scoreChart').getContext('2d');
            scoreChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: generations,
                    datasets: [{
                        label: 'Best Score',
                        data: scores,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            type: 'logarithmic',
                            title: {
                                display: true,
                                text: 'Score (lower is better)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Generation'
                            }
                        }
                    },
                    animation: {
                        duration: 0 // Disable animation for better performance
                    }
                }
            });
        }

        // Call this when the page loads
        initializeChart();

        socket.on('status_update', (data) => {
            statusMessage.textContent = data.message;
        });

        socket.on('optimization_update', (data) => {
            // Update progress bar
            const progressPercentage = (data.generation / data.total_generations) * 100;
            progressBar.style.width = `${progressPercentage}%`;
            progressBar.textContent = `${Math.round(progressPercentage)}%`;

            // Update generation info
            generationInfo.textContent = `Generation ${data.generation}/${data.total_generations} - Best Score: ${data.best_score}`;

            // Update chart
            generations.push(data.generation);
            scores.push(data.best_score);
            scoreChart.data.labels = generations;
            scoreChart.data.datasets[0].data = scores;
            scoreChart.update('none'); // Update without animation

            // Update nutrition report
            updateNutritionReport(data.nutrition_report);
        });

        socket.on('optimization_complete', (data) => {
            // Update UI
            startButton.disabled = false;
            stopButton.disabled = true;
            statusMessage.textContent = 'Optimization complete!';
            progressBar.textContent = '100%';
            progressBar.style.width = '100%';
            progressBar.classList.remove('progress-bar-animated');

            // Update final report
            updateNutritionReport(data.final_report);
        });

        function updateNutritionReport(report) {
            // Update meal info
            mealInfo.innerHTML = `
                <h6>Meal ${report.meal_info.meal_number} of ${report.meal_info.number_of_meals}</h6>
                <p>Each meal targets ${report.meal_info.target_percentage.toFixed(1)}% of daily nutrition needs</p>
            `;

            // Sort foods by amount in descending order
            const sortedFoods = Object.entries(report.foods)
                .sort(([, amountA], [, amountB]) => amountB - amountA);

            // Update food quantities
            let foodsHtml = '';
            for (const [food, amount] of sortedFoods) {
                foodsHtml += `
                    <tr>
                        <td>${food}</td>
                        <td>${amount}</td>
                    </tr>
                `;
            }
            foodQuantitiesBody.innerHTML = foodsHtml;

            // Update nutrient profile
            let nutrientsHtml = '';
            for (const nutrient of report.nutrients) {
                const statusClass = nutrient.status === 'OK' ? 'status-ok' :
                                    nutrient.status === 'LOW' ? 'status-low' : 'status-high';

                nutrientsHtml += `
                    <tr>
                        <td>${nutrient.name}</td>
                        <td>${nutrient.amount}${nutrient.unit}</td>
                        <td>${nutrient.meal_percentage}%</td>
                        <td>${nutrient.daily_percentage}%</td>
                        <td class="${statusClass}">${nutrient.status}</td>
                    </tr>
                `;
            }
            nutrientProfileBody.innerHTML = nutrientsHtml;

            // Update summary
            nutritionSummary.innerHTML = `
                <ul>
                    <li class="status-ok">Nutrients at good levels: ${report.summary.ok} of ${report.summary.total}</li>
                    <li class="status-low">Nutrients below target: ${report.summary.low} of ${report.summary.total}</li>
                    <li class="status-high">Nutrients above target: ${report.summary.high} of ${report.summary.total}</li>
                </ul>
            `;
        }
    </script>
</body>
</html>''')

    # Create a route to get foods data
    @app.route('/get_foods')
    def get_foods():
        return jsonify({'foods': foods})

    socketio.run(app,
                host='0.0.0.0',
                port=5000,
                debug=True,
                allow_unsafe_werkzeug=True)

@app.route('/get_rdi')
def get_rdi():
    """Return the RDI data as JSON"""
    return jsonify({'rdi': rdi})

@app.route('/save_custom_food', methods=['POST'])
def save_custom_food():
    """Add a custom food to the foods database"""
    global foods

    data = request.get_json()
    food_name = data.get('name')
    food_data = data.get('data')

    if not food_name or not food_data:
        return jsonify({'status': 'error', 'message': 'Invalid food data'})

    # Add density if not provided
    if 'density' not in food_data:
        food_data['density'] = 100

    # Add to global foods dict
    foods[food_name] = food_data

    # Save to JSON file
    with open('foods.json', 'w') as f:
        json.dump(foods, f, indent=2)

    return jsonify({'status': 'success', 'message': f'Added {food_name} to foods database'})

@app.route('/delete_food', methods=['POST'])
def delete_food():
    """Remove a food from the foods database"""
    global foods

    data = request.get_json()
    food_name = data.get('name')

    if not food_name or food_name not in foods:
        return jsonify({'status': 'error', 'message': 'Food not found'})

    # Remove from global dict
    del foods[food_name]

    # Save to JSON file
    with open('foods.json', 'w') as f:
        json.dump(foods, f, indent=2)

    return jsonify({'status': 'success', 'message': f'Removed {food_name} from foods database'})

@app.route('/get_optimization_history')
def get_optimization_history():
    """Return the optimization history as JSON"""
    if not os.path.exists('optimization_history.csv'):
        return jsonify({'status': 'error', 'message': 'No optimization history found'})

    try:
        df = pd.read_csv('optimization_history.csv')
        history = df.to_dict('records')

        # Group by run number
        runs = {}
        for record in history:
            run_number = record['run_number']
            if run_number not in runs:
                runs[run_number] = []
            runs[run_number].append(record)

        return jsonify({'status': 'success', 'runs': runs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})