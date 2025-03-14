import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Parameters for the genetic algorithm
population_size = 50
generations = 100
mutation_rate = 0.1
target_value = 0

def initialize_population():
    return np.random.uniform(-100, 100, population_size)

def fitness(population):
    return np.abs(population - target_value)

def selection(population):
    fitness_scores = fitness(population)
    sorted_indices = np.argsort(fitness_scores)
    return population[sorted_indices[:population_size // 2]]

# Simplified crossover for single values
def crossover(parent1, parent2):
    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def mutation(offspring):
    if np.random.rand() < mutation_rate:
        offspring += np.random.uniform(-10, 10)
    return offspring

def run_genetic_algorithm():
    population = initialize_population()
    all_populations = [population]

    for gen in range(generations):
        selected = selection(population)

        # Create new population via crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutation(child1), mutation(child2)])

        population = np.array(new_population)
        all_populations.append(population)

    return all_populations

def animate_ga():
    all_populations = run_genetic_algorithm()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, population_size)

    def update(frame):
        ax.clear()
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, population_size)

        population = all_populations[frame]
        fitness_scores = fitness(population)
        y_positions = np.linspace(0, population_size, len(population))  # Create evenly spaced y-positions

        scatter = ax.scatter(population, y_positions,
                           c=fitness_scores,
                           cmap='viridis',
                           s=50)

        ax.set_title(f"Generation: {frame}/{generations}")
        ax.set_xlabel("Individual Value")
        ax.set_ylabel("Population Index")

        # Add a colorbar if it doesn't exist
        if not hasattr(fig, 'colorbar'):
            fig.colorbar(scatter, ax=ax, label='Fitness Score')

    ani = FuncAnimation(fig, update, frames=generations + 1,
                       repeat=True, interval=100)  # 100ms between frames

    # Save as GIF
    gif_file = "genetic_algorithm_animation.gif"
    print(f"Saving animation to {gif_file}...")

    writer = animation.PillowWriter(fps=10)
    ani.save(gif_file, writer=writer)
    plt.close()

if __name__ == "__main__":
    animate_ga()