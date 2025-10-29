import matplotlib.pyplot as plt
from genome_ps2 import run_ga_with_selection_tracking, report_final_solution

# this entire file was used to practice matplotlib
# im sure there are easier ways to graph these values

def plot_selection_comparison(config_file, mutation_rate=0.1, save_path=None):
    methods = ["roulette", "tournament"]
    results = {}

    for method in methods:
        avg_fit, best_chrom, best_fit, best_gen, W, weights, values = run_ga_with_selection_tracking(
            config_file, Mr=mutation_rate, selection_method=method)
        
        results[method] = {"avg": avg_fit,"best_fit": best_fit,"best_gen": best_gen,"best_chrom": best_chrom,
            "W": W, "weights": weights, "values": values}

    # plot average fitness across generations for both methods
    plt.figure(figsize=(8, 5))
    for method in methods:
        plt.plot(results[method]["avg"], label=f"{method.capitalize()} Selection")

    plt.title(f"Average Fitness per Generation — {config_file}")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # determine default save path if not provided
    if save_path is None:
        base = config_file.replace(".txt", "").replace(" ", "_")
        save_path = f"{base}_selection_comparison.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # print best solutions summary for both methods
    print(f"\n=== {config_file} — Best Solutions ===")
    for method in methods:
        print(f"\n[{method.capitalize()}]")
        print(f"Best Fitness: {results[method]['best_fit']} (Gen {results[method]['best_gen']})")
        report_final_solution(results[method]["best_chrom"], results[method]["W"], results[method]["weights"],results[method]["values"])
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    # generate plots for both config files with default mutation rate
    plot_selection_comparison("config_1.txt", mutation_rate=0.1, save_path="ga_config_1_selection_comparison.png")
    plot_selection_comparison("config_2.txt", mutation_rate=0.1, save_path="ga_config_2_selection_comparison.png")