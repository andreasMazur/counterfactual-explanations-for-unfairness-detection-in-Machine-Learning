from compute_cfs import run_experiment
from visualization import create_plots
from chi2_tests import compute_chi2_for_groups


def main():
    """
    Runs the entire experiment.
    """
    # Run the experiment
    answer = input("Conceal sensitive attributes? (y/n)")
    if answer == "y":
        conceal = True
    else:
        conceal = False
    run_experiment(conceal)

    # Plot figures
    create_plots()

    # Compute chi2-test statistics
    if not conceal:
        compute_chi2_for_groups()


if __name__ == "__main__":
    main()
