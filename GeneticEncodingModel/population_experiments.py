from __future__ import annotations

import numpy as np
import pandas as pd

# tried to run this in the original file and broke it several times and got frustrated
# so it was just easier to import over and run seperate

from genome_ps2 import (calculate_population_fitness, crossover_operation, get_best_solution_metrics,
    get_initial_population, get_selection_probabilities, mutation_operation, select_parents, tournament_selection)


DEFAULT_POPULATION_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def run_ga_trial( population_size: int, num_items: int, capacity: int, weights: np.ndarray,
    values: np.ndarray, num_generations: int, mutation_rate: float,selection_method: str) -> tuple[float, float, int]:

    # executes one GA run and returns (best_value, best_weight, items_selected).

    if population_size % 2 != 0:
        raise ValueError("population_size must be even to support paired crossover.")

    # randomly initialise the generation-0 population for this trial.
    population = np.random.randint(2, size=(population_size, num_items))
    generation = 0
    best_fitness = 0.0
    best_chromosome = population[0].copy()

    while generation < num_generations:
        fitness = calculate_population_fitness(population, capacity, weights, values)

        # track the best feasible chromosome seen so far.
        best_index = int(np.argmax(fitness))
        if fitness[best_index] > best_fitness:
            best_fitness = float(fitness[best_index])
            best_chromosome = population[best_index].copy()

        # selection
        if selection_method == "tournament":
            mating_pool = tournament_selection(population, fitness)
        else:
            selection_prob = get_selection_probabilities(fitness)
            mating_pool = select_parents(population, selection_prob)

        # variation operators
        offspring = crossover_operation(mating_pool, population.shape[1])
        population = mutation_operation(offspring, mutation_rate)

        generation += 1

    value, weight, items = get_best_solution_metrics(best_chromosome, capacity, weights, values)
    return float(value), float(weight), int(items)


def run_population_experiments(config_file: str = "config_1.txt", population_sizes: list[int] | None = None,
    num_trials: int = 30, mutation_rate: float = 0.1, selection_method: str = "roulette", num_generations: int | None = None,
    random_seed: int | None = 0,) -> pd.DataFrame:
    # executes the GA for a range of population sizes and aggregates statistics.
    # returns a dataframe with per-population mean / std for value, weight, and
    # number of included items, as well as maxima needed for reporting.

    base_population, capacity, raw_items, _, stop = get_initial_population(config_file)  # initialize population and load problem variables
    num_items = base_population.shape[1] # extract the number of items (columns in the population matrix)
    items_array = np.array(raw_items, dtype=int) # convert item list of tuples into a numpy array for easy slicing
    weights = items_array[:, 0] # extract the weights from the first column
    values = items_array[:, 1] # extract the values from the second column

    max_generations = stop if num_generations is None else min(stop, num_generations) # determine the total number of generations to run
    populations = population_sizes or DEFAULT_POPULATION_SIZES # set list of population sizes (use default if not specified)

    summaries = [] # list to store results from each population size experiment

    for idx, pop_size in enumerate(populations): # iterate over all selected population sizes
        final_values = [] # store fitness values from each trial
        final_weights = [] # store total weight from each trial
        final_item_counts = [] # store number of selected items from each trial

        for trial in range(num_trials): # run multiple trials for statistical reliability
            if random_seed is None:
                np.random.seed() # use system-based random seed for variation
            else:
                np.random.seed(random_seed + idx * 10_000 + trial) # vary seed based on population and trial index

            value, weight, items = run_ga_trial(population_size=pop_size, num_items=num_items,
                capacity=capacity, weights=weights, values=values, num_generations=max_generations,
                mutation_rate=mutation_rate, selection_method=selection_method) # run a single GA trial and get results

            final_values.append(value) # append best value from this trial
            final_weights.append(weight) # append total weight from this trial
            final_item_counts.append(items) # append number of items in the best solution

        summaries.append( # store statistical summaries for each population size
            {   "population_size": pop_size,
                "value_mean": float(np.mean(final_values)), # mean fitness
                "value_std": float(np.std(final_values)), # std deviation of fitness
                "value_min": float(np.min(final_values)), # min fitness observed
                "value_max": float(np.max(final_values)), # max fitness observed
                "weight_mean": float(np.mean(final_weights)), # mean total weight
                "weight_std": float(np.std(final_weights)), # std deviation of weight
                "weight_max": float(np.max(final_weights)), # max weight observed
                "items_mean": float(np.mean(final_item_counts)), # mean number of items selected
                "items_std": float(np.std(final_item_counts)), # std deviation of item count
                "items_max": float(np.max(final_item_counts)), # max number of items selected
            }
        )

    summary_df = pd.DataFrame(summaries)  # convert the summaries list into a dataframe for easier reporting
    return summary_df[  # select only relevant columns for clean output
        [
            "population_size",
            "value_mean",
            "value_std",
            "value_min",
            "value_max",
            "weight_mean",
            "weight_std",
            "weight_max",
            "items_mean",
            "items_std",
            "items_max",
        ]
    ]


if __name__ == "__main__":
    summary = run_population_experiments()  # execute the main experiment when file is run directly
    print("\nPopulation Size Sweep ({} trials per setting)\n".format(30))  # header for output display
    try:
        print(summary.to_markdown(index=False, floatfmt=".2f"))  # print formatted table if markdown is available
    except Exception:
        # fallback if tabulate is unavailable.
        print(summary.to_string(index=False, float_format="{:.2f}".format))  # plain text output if markdown fails