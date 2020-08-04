from compute_cfs import run_experiment
from visualization import create_plots
from chi2_tests import compute_chi2_for_groups


def main():
    # Run the experiment
    run_experiment()

    # Plot figures
    create_plots()

    # Compute chi2-test statistics
    compute_chi2_for_groups()


if __name__ == "__main__":
    main()
