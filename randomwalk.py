import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_winrate(points):
    """
    Calculate winrate based on current points.
    At 0 points: 50%
    At 100 points: 15%
    At -100 points: 85% (symmetric)
    Linear interpolation between these points.
    """
    # Linear model: winrate = 50% - (points * 0.35/100)
    winrate = 0.5 - (points * 0.0035)

    # Clamp between 0.01 and 0.99 to avoid impossible probabilities
    return max(0.01, min(0.99, winrate))


def simulate_single_run(target=100, max_negative=-200, max_steps=10000):
    """
    Simulate a single run starting from 0 points.

    Returns:
    - 'success' if reached target
    - 'failure_negative' if went below max_negative
    - 'failure_timeout' if exceeded max_steps
    """
    points = 0
    steps = 0

    while steps < max_steps:
        # Check win conditions
        if points >= target:
            return 'success'
        if points <= max_negative:
            return 'failure_negative'

        # Calculate current winrate and make a move
        winrate = get_winrate(points)
        if random.random() < winrate:
            points += 1  # Win
        else:
            points -= 1  # Loss

        steps += 1

    return 'failure_timeout'


def run_simulation(num_runs=100000, target=100, verbose=True):
    """
    Run multiple simulations and calculate statistics.
    """
    results = defaultdict(int)

    if verbose:
        print(f"Running {num_runs:,} simulations...")
        print(f"Target: {target} points")
        print(f"Winrate function: 50% at 0, 15% at {target}, 85% at -{target}")
        print("-" * 50)

    for i in range(num_runs):
        if verbose and (i + 1) % 10000 == 0:
            print(f"Completed {i + 1:,} runs...")

        result = simulate_single_run(target)
        results[result] += 1

    # Calculate probabilities
    total_runs = sum(results.values())
    prob_success = results['success'] / total_runs
    prob_failure_neg = results['failure_negative'] / total_runs
    prob_timeout = results['failure_timeout'] / total_runs

    if verbose:
        print(f"\nResults after {total_runs:,} runs:")
        print(
            f"Success (reached {target}): {results['success']:,} ({prob_success:.4%})")
        print(
            f"Failed (too negative): {results['failure_negative']:,} ({prob_failure_neg:.4%})")
        print(f"Timed out: {results['failure_timeout']:,} ({prob_timeout:.4%})")
        print(
            f"\nEstimated probability of reaching {target}: {prob_success:.4%}")

    return prob_success, results


def plot_winrate_function():
    """
    Plot the winrate function to visualize the bias.
    """
    points = np.linspace(-150, 150, 301)
    winrates = [get_winrate(p) for p in points]

    plt.figure(figsize=(10, 6))
    plt.plot(points, winrates, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% line')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=100, color='g', linestyle='--', alpha=0.7,
                label='Target (100)')
    plt.xlabel('Points')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Current Points')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1)

    # Add annotations
    plt.annotate('50% at 0', (0, 0.5), xytext=(20, 0.6),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate('15% at 100', (100, 0.15), xytext=(80, 0.3),
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.show()


def sensitivity_analysis():
    """
    Test how the probability changes with different numbers of simulation runs.
    """
    run_counts = [1000, 5000, 10000, 25000, 50000, 100000]
    probabilities = []

    print("Sensitivity Analysis - How estimate changes with more runs:")
    print("Runs\t\tProbability\tChange")
    print("-" * 40)

    prev_prob = None
    for runs in run_counts:
        prob, _ = run_simulation(runs, verbose=False)
        probabilities.append(prob)

        change_str = ""
        if prev_prob is not None:
            change = abs(prob - prev_prob)
            change_str = f"{change:.4%}"

        print(f"{runs:,}\t\t{prob:.4%}\t\t{change_str}")
        prev_prob = prob


if __name__ == "__main__":
    # Show the winrate function
    plot_winrate_function()

    # Run main simulation
    probability, results = run_simulation(num_runs=100000)

    # Run sensitivity analysis
    print("\n" + "=" * 60)
    sensitivity_analysis()

    # Quick theoretical note
    print("\n" + "=" * 60)
    print("THEORETICAL NOTE:")
    print("This is a biased random walk with negative drift.")
    print("The probability of reaching +100 should be quite low")
    print("because the winrate decreases as we get higher.")