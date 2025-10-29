import numpy as np
import pandas as pd
import tempfile
import os

def _load_config_lines(path):
    with open(path, "r") as f:
        return f.readlines()

def _write_temp_config_with_popsize(base_config_path, pop_size):
    # temporarily override the population size in the first line of a config file.
    lines = _load_config_lines(base_config_path)
    lines[0] = str(int(pop_size)) + "\n"   # override pop_size only
    tf = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
    tf.writelines(lines)
    tf.close()
    return tf.name

def get_initial_population(config_1):
    np.random.seed(1470)  # manages reproducibility

    # read all lines
    with open(config_1, "r") as file:
        lines = file.readlines()
    pop_size, n, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]] #mapping the values onto a tuple, taking out the newline
    # then splitting then into their variables

    #initialize g and P here
    g = 0

    P = np.random.randint(2, size=(pop_size, n)) #initializing population P at generation g=0

    return P, W, S, g, stop

def calculate_fitness(C, W, weights, values):
    total_w = np.sum(C * weights)
    total_v = np.sum(C * values)

    if total_w <= W:
        return total_v
    else:
        return 0

def calculate_population_fitness(P, W, weights, values): #calculate total value and total weight for all individuals
    total_values = np.dot(P, values) #shape pop_size
    total_weights = np.dot(P, weights) #shape pop_size

    # initilaize the array for fitness score
    fitness = np.zeros(P.shape[0])

    # implement the contstraint where total_weights <=W and apply the logic to all elements in the matrix
    fitness = np.where(total_weights <= W, total_values, 0)
    
    return fitness

def get_selection_probabilities(fitness):
    total_fitness = np.sum(fitness)

    if total_fitness == 0: # if all fitness is zero, give equal probabilies to all individuals
        pop_size = len(fitness)
        selection_prob = np.ones(pop_size) / pop_size #preventing a divide by 0 issue just in case sum of all fitness is 0
    
    else:
        # calculate the probability for each individual
        selection_prob = fitness / total_fitness
    
    return selection_prob

def select_parents(P, selection_prob): # passing P and the probabilities
    pop_size = P.shape[0] # get pop_size from P

    parent_indicies = np.random.choice(a=np.arange(pop_size), size=(pop_size), p=selection_prob, replace=True)
    # to select randomly, make and array of n, n+1,...,n-1 and set it to pop_size
    # then take the probability distribution for choice, then replace=True allows the same individual to be selected twice

    #use the indecies to select the chromosome from P
    P_mating = P[parent_indicies, :]

    return P_mating

def tournament_selection(P, fitness, tournament_size=3):
    
    # implements tournament selection.
    # randomly selects 'tournament_size' individuals and picks the fittest one.

    pop_size = P.shape[0]
    selected = []

    # prevent invalid sampling when population < tournament_size
    t_size = min(tournament_size, pop_size)

    for _ in range(pop_size):
        indices = np.random.choice(pop_size, t_size, replace=False)
        best_index = indices[np.argmax(fitness[indices])]
        selected.append(P[best_index])

    return np.array(selected)


def crossover_operation(P_mating, n):
    pop_size = P_mating.shape[0]
    offspring = [] # list to hold the new chromosomes

    # iterating on the mating pool, select parents 2 at a time
    for i in range(0, pop_size, 2):
        parent_1 = P_mating[i, :] # first parent
        parent_2 = P_mating[i+1, :] # second parent

        crossover_point = np.random.randint(1, n) # choose the crossover point

        offspring_1 = np.concatenate([parent_1[0:crossover_point], parent_2[crossover_point:n]])
        offspring_2 = np.concatenate([parent_2[0:crossover_point], parent_1[crossover_point:n]])
        # parent_1[0:crossover_point] indicies from 0 up to the "cut"
        # parent_2[0:crossover_point] inficies from "cut" to the end
        # then visa versa

        offspring.append(offspring_1) # add offspring to the list
        offspring.append(offspring_2)

    P_offspring = np.vstack(offspring) # convert the list of offspring into 1 matrix

    return P_offspring

def mutation_operation(P_offspring, Mr):
    random_matrix = np.random.rand(*P_offspring.shape) # creating the random probability matrix
    mutation_mask = (random_matrix < Mr) # boolean mask (True = mutate, False = dont) also identifies where the mutation occurs
    P_offspring[mutation_mask] = 1 - P_offspring[mutation_mask] # selects only the gene that needs a mutation
    # 1 - P_offspring[mutation_mask] flips a zero to a 1, and a 1 to a 0

    return P_offspring

def solve_knapsack_ga(config_file, Mr): # had a lot of issues in this entire def section
    P, W, S, g, stop = get_initial_population(config_file)

    S_array = np.array(S) # data prep steps
    weights = S_array[:, 0]
    values = S_array[:, 1]

    best_overall_fitness = 0
    best_chromosome = P[0, :].copy() # initialize with a copy of the first chromosome
    best_generation = 0

    while g < stop:
        # evaluation and selection
        fitness = calculate_population_fitness(P, W, weights, values) #calculate the fitness of the current population P

        if g == 0 and "config_2" in config_file:
        # skip evaluation entirely on first generation
            print(f"Initial population skipped for config_2; forcing first reproduction...")

        # equal probability selection (no fitness bias)
            selection_prob = np.ones(P.shape[0]) / P.shape[0]
            P_mating = select_parents(P, selection_prob)

        # standard GA reproduction process
            P_offspring = crossover_operation(P_mating, P.shape[1])
            P = mutation_operation(P_offspring, Mr)
            g += 1

        # recalculate fitness for new generation
        fitness = calculate_population_fitness(P, W, weights, values)

        print(f"Generation {g}: feasible individuals = {np.sum(np.dot(P, weights) <= W)} / {P.shape[0]}")


        current_max_fitness = np.max(fitness) #track before the crossover ===> everything under this line was bugged
        if current_max_fitness > best_overall_fitness:
            best_overall_fitness = current_max_fitness #a new best solution was found, update the tracking variables
            best_index = np.argmax(fitness) #find the index of the chromosome that achieved this max fitness
            best_chromosome = P[best_index, :].copy() #save the unmutated chromosome
            best_generation = g

        selection_prob = get_selection_probabilities(fitness) #calculating the selection probabilities
        P_mating = select_parents(P, selection_prob) #getting the mating pool
       
        #genetic operations
        P_offspring = crossover_operation(P_mating, P.shape[1]) #getting the offspring
        P = mutation_operation(P_offspring, Mr) #mutate the offspring
        g += 1 #increment generation counter

    print(f"--- GA Run complete ---")
    print(f"You are now a Biologist")
    print(f"Final Generation Reached {g}")
    print(f"Best Fitness Overall: {best_overall_fitness} (Found in Generation:{best_generation})")

    return P, best_chromosome, best_overall_fitness, best_generation, W, weights, values

def solve_knapsack_selection_only(config_file, selection_method="roulette"):

    # Runs GA with only selection active.
    # selection_method: "roulette" or "tournament"
    P, W, S, g, stop = get_initial_population(config_file)
    S_array = np.array(S)
    weights = S_array[:, 0]
    values = S_array[:, 1]

    avg_fitness_per_gen = [] #storing values of each generation
    best_fitness_per_gen = []
    active_genes_per_gen = []

    best_overall_fitness = 0
    best_chromosome = None
    best_generation = 0

    while g < stop:
        fitness = calculate_population_fitness(P, W, weights, values)

        # Track averages
        avg_fitness_per_gen.append(np.mean(fitness))
        best_idx = np.argmax(fitness)
        best_fitness_per_gen.append(fitness[best_idx])
        active_genes_per_gen.append(np.sum(P[best_idx]))

        if fitness[best_idx] > best_overall_fitness:
            best_overall_fitness = fitness[best_idx]
            best_chromosome = P[best_idx].copy()
            best_generation = g

        # Selection only
        if selection_method == "roulette":
            selection_prob = get_selection_probabilities(fitness)
            P = select_parents(P, selection_prob)
        elif selection_method == "tournament":
            P = tournament_selection(P, fitness)
        else:
            raise ValueError("Unknown selection method")

        g += 1

    # Final results
    print(f"\n--- Selection Only ({selection_method}) for {config_file} ---")
    print(f"Best Fitness Overall: {best_overall_fitness}")
    print(f"Found in Generation: {best_generation}")
    print(f"Active Genes: {np.sum(best_chromosome)}")

    return avg_fitness_per_gen, best_fitness_per_gen, active_genes_per_gen, best_overall_fitness, best_generation


def get_best_solution_metrics(C, W, weights, values):
    """Calculates metrics for the best chromosome for tabular reporting."""
    total_v = np.sum(C * values)
    total_w = np.sum(C * weights)
    active_genes = np.sum(C)
    
    # Fitness is the total value if feasible, 0 otherwise.
    fitness = total_v if total_w <= W else 0
    
    return fitness, total_w, active_genes

def report_final_solution(C, W, weights, values):
    # Calculates and prints the key metrics for a single chromosome C.
    fitness, total_w, active_genes = get_best_solution_metrics(C, W, weights, values)
    
    print(f"Total Value (Fitness): {fitness}")
    print(f"Total Weight: {total_w} (Max Capacity: {W})")
    print(f"Number of Items (Active Genes): {active_genes}")
    
    # Calculate feasibility based on the strict rule (<= W)
    if total_w <= W:
        print("Status: Feasible (Constraint Met)")
    else:
        print("Status: Infeasible (Weight Constraint Violated)")

#Main Execution Block

def run_ga_with_selection_tracking(config_file, Mr=0.1, selection_method="roulette"):
    # Runs GA including selection, crossover, and mutation while tracking average fitness.
    # selection_method: "roulette" or "tournament".
    # Returns (avg_fitness_per_gen, best_chromosome, best_overall_fitness, best_generation, W, weights, values)

    P, W, S, g, stop = get_initial_population(config_file)
    S_array = np.array(S)
    weights = S_array[:, 0]
    values = S_array[:, 1]

    avg_fitness_per_gen = []
    best_overall_fitness = 0
    best_chromosome = P[0, :].copy()
    best_generation = 0

    while g < stop:
        fitness = calculate_population_fitness(P, W, weights, values)

        # Track averages and best
        avg_fitness_per_gen.append(np.mean(fitness))
        current_max_fitness = np.max(fitness)
        if current_max_fitness > best_overall_fitness:
            best_overall_fitness = current_max_fitness
            best_index = np.argmax(fitness)
            best_chromosome = P[best_index, :].copy()
            best_generation = g

        # Selection step based on method
        if g == 0 and "config_2" in config_file:
            # Special handling for config_2 first generation
            selection_prob = np.ones(P.shape[0]) / P.shape[0]
            P_mating = select_parents(P, selection_prob)
        else:
            if selection_method == "roulette":
                selection_prob = get_selection_probabilities(fitness)
                P_mating = select_parents(P, selection_prob)
            elif selection_method == "tournament":
                P_mating = tournament_selection(P, fitness)
            else:
                raise ValueError("Unknown selection method: " + str(selection_method))

        # Genetic operators
        P_offspring = crossover_operation(P_mating, P.shape[1])
        P = mutation_operation(P_offspring, Mr)
        g += 1

    return (avg_fitness_per_gen, best_chromosome, best_overall_fitness, best_generation, W,
        weights, values,)

def run_comparison_experiment(config_file_1, config_file_2, Mr=0.1):
    
    # Runs the Genetic Algorithm for two configuration files and outputs a comparison table.
    results = {}
    
    # Run File 1
    print(f"\n--- Starting GA Run for: {config_file_1} ---")
    
    P_final_1, C_best_1, fitness_best_1, gen_best_1, W1, weights1, values1 = solve_knapsack_ga(config_file_1, Mr)
    
    # Get all the metrics for the report
    fitness_1, weight_1, items_1 = get_best_solution_metrics(C_best_1, W1, weights1, values1)
    
    results[config_file_1] = {'Capacity': W1, 'Value': fitness_1, 'Weight': weight_1,
        'Items': items_1, 'Generation Found': gen_best_1}
    
    # Prompt for File 2
    while True:
        run_next = input(f"Do you want to run the next configuration file, {config_file_2}? (y/n): ")
        if run_next.lower() == 'y':
            break
        elif run_next.lower() == 'n':
            print("Stopping experiment. Comparison table will only show one result.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Run File 2 If requested
    if len(results) == 1 and run_next.lower() == 'y':
        print(f"\n--- Starting GA Run for: {config_file_2} ---")
        
        P_final_2, C_best_2, fitness_best_2, gen_best_2, W2, weights2, values2 = solve_knapsack_ga(config_file_2, Mr)

        fitness_2, weight_2, items_2 = get_best_solution_metrics(C_best_2, W2, weights2, values2)

        results[config_file_2] = {
            'Capacity': W2,
            'Value': fitness_2,
            'Weight': weight_2,
            'Items': items_2,
            'Generation Found': gen_best_2
        }

    # Generate Comparison Table
    print("\n" + "="*70)
    print("                Genetic Algorithm Performance Comparison")
    print("="*70)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    # Use Markdown format for clean output in consoles/reports
    print(df.to_markdown())
    
    print(f"Experiment complete.")

# Population Size Experiment
def run_q4_population_sweep(
    base_config_path,
    pop_sizes=(2,4,6,8,10,20,40,60,80,100),
    trials=30,
    mutation_rate=0.10,
    selection_method="roulette"  # or "tournament"
):
    
    # Runs GA for multiple population sizes and reports mean ± sd fitness, max weight, and average items.
    # Matches the Table 1 format required for Q4.

    rows = []
    for ps in pop_sizes:
        tmp_cfg = _write_temp_config_with_popsize(base_config_path, ps)
        final_values, final_weights, final_items = [], [], []
        try:
            for _ in range(trials):
                # Reuse your existing full GA with selection, crossover, and mutation
                avg_fit, C_best, best_fit, gen_best, W, weights, values = run_ga_with_selection_tracking(
                    tmp_cfg, Mr=mutation_rate, selection_method=selection_method
                )
                fit, wt, items = get_best_solution_metrics(C_best, W, weights, values)
                final_values.append(fit)
                final_weights.append(wt)
                final_items.append(items)
        finally:
            os.unlink(tmp_cfg)

        rows.append({
            "Pop Size": ps,
            "Knapsack Total Value (x ± sd)": f"{np.mean(final_values):.2f} ± {np.std(final_values):.2f}",
            "Knapsack Max Weight": int(np.max(final_weights)),
            "# Items in Best Solution": float(np.mean(final_items)),})

    df = pd.DataFrame(rows).sort_values("Pop Size")
    print("\n" + "="*70)
    print("                Q4: Genetic Algorithm Population Size Sweep")
    print("="*70)
    print(df.to_markdown(index=False))
    return df


#Execution Block
if __name__ == '__main__':
    FILE_1 = 'config_1.txt'
    FILE_2 = 'config_2.txt'
    MUTATION_RATE = 0.1

    # PHASE 1: Full GA Run
    run_comparison_experiment(FILE_1, FILE_2, MUTATION_RATE)

    # Prompt for PHASE 2
    proceed = input(f"Do you want to run Selection-Only experiments now? (y/n): ")
    if proceed.lower() == 'y':
        print("\n" + "="*70)
        print("           Running Selection-Only Experiments            ")
        print("="*70)

        for file in [FILE_1, FILE_2]:
            for method in ["roulette", "tournament"]:
                avg_fit, best_fit, active_genes, best_overall, gen_best = solve_knapsack_selection_only(file, method)
                print(f"\n--- Selection Only ({method}) for {file} ---")
                print(f"Best Fitness Overall: {best_overall}")
                print(f"Found in Generation: {gen_best}")
                print(f"Active Genes: {active_genes[-1] if len(active_genes) > 0 else 'None'}")
    else:
        print(f"Selection-only phase skipped.")

    # Optional Experiment
    q4_run = input(f"Run Q4 population size experiment? (y/n): ")
    if q4_run.lower() == 'y':
        print(f"Running population sweep for Q4...")
        df_roulette = run_q4_population_sweep(
            "config_1.txt",
            pop_sizes=[2,4,6,8,10,20,40,60,80,100],
            trials=30,
            mutation_rate=0.10,
            selection_method="roulette")
        
        df_tournament = run_q4_population_sweep(
            "config_1.txt",
            pop_sizes=[2,4,6,8,10,20,40,60,80,100],
            trials=30,
            mutation_rate=0.10,
            selection_method="tournament")
        
    else:
        print(f"Q4 experiment skipped.")