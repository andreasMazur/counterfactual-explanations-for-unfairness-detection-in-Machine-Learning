from compute_cfs import run_experiment
from visualization import create_plots
from chi2_tests import compute_chi2_for_groups


def main():
    """
    Runs the entire experiment.
    """

    conceal = input("Conceal sensitive attributes? (y/n)")
    if conceal == "y":
        conceal_attributes = True
    else:
        conceal_attributes = False

    # Run the experiment
    run_experiment(conceal_attributes)

    # Plot figures
    create_plots()

    # Compute chi2-test statistics
    if not conceal_attributes:
        compute_chi2_for_groups()


if __name__ == "__main__":
    main()
