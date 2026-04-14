import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import BaseModel

from rag_hpo_bench.data_models.data_sampling_params import DataSamplingParams
from rag_hpo_bench.data_models.dataset_id import DatasetID
from rag_hpo_bench.data_models.dataset_names import DatasetName
from rag_hpo_bench.hpo.hpo_algorithm import HpoAlgorithmType
from rag_hpo_bench.hpo.hpo_experiment import HpoExperiment
from rag_hpo_bench.hpo.hpo_results import HpoResults
from rag_hpo_bench.hpo.pattern_results import MultiSeedTestResults
from rag_hpo_bench.hpo.test_results import TestResults
from rag_hpo_bench.utils.logging_utils import init_logger

logger = logging.getLogger(__name__)

SamplingSetupType = Literal["sample", "full"]
SplitType = Literal["Dev", "Test"]


class MetricDefinition(BaseModel):
    """Definition of a metric with its various name representations."""

    name: str  # Internal name used in code
    display_name: str  # Human-readable name for display
    short_name: str  # Abbreviated name for plots/tables


# Define the three metrics used in analysis
LLMAAJ_AC = MetricDefinition(name="LLMaaJ-AC", display_name="LLMaaJ-AC", short_name="AC")

LEXICAL_AC = MetricDefinition(name="Lexical-AC", display_name="Lexical-AC", short_name="LAC")

LEXICAL_FF = MetricDefinition(name="Lexical-FF", display_name="Lexical-FF", short_name="LF")


def get_sampling_params(sampling_setup: str, dataset_name: DatasetName) -> DataSamplingParams:
    """Get sampling parameters based on setup type and dataset.

    Args:
        sampling_setup: Type of sampling setup ("sample", "full", or specific like "max_150_questions")
        dataset_name: Dataset name enum

    Returns:
        DataSamplingParams object with appropriate settings
    """
    if sampling_setup == "full":
        return DataSamplingParams()
    elif sampling_setup == "sample":
        # Default sample configuration
        return DataSamplingParams()
    elif sampling_setup == "max_150_questions":
        return DataSamplingParams()
    else:
        # Try to parse as a custom configuration
        return DataSamplingParams()


def get_algorithm_setups() -> list[HpoAlgorithmType]:
    """Get list of algorithm setups to analyze.

    Returns:
        List of HpoAlgorithmType values included in test result analysis
    """
    return [
        HpoAlgorithmType.RANDOM,
        HpoAlgorithmType.GREEDY_M,
        HpoAlgorithmType.GREEDY_R,
    ]


def get_greedy_algorithm_setups(
    metric_short_name: str, with_sampling: bool
) -> list[tuple[str, str]]:
    """Get list of greedy algorithm setups.

    Args:
        metric_short_name: Short name of the metric
        with_sampling: Whether sampling is enabled

    Returns:
        List of tuples (display_name, algorithm_id)
    """
    # Return empty list for now - greedy algorithms would be added here
    return []


def get_grid_results(
    dataset_name: DatasetName,
    split: SplitType,
    base_results_path: Path,
) -> HpoResults:
    """Load grid search results for a specific configuration.

    Args:
        dataset_name: Dataset name enum
        split: Data split (Dev or Test)
        base_results_path: Base path for experiment results

    Returns:
        HpoResults object containing the grid search results
    """
    # Create DatasetID for the tune dataset
    tune_dataset = DatasetID(
        dataset_name=dataset_name,
        split=split,
        sampling_params=DataSamplingParams(),
    )

    grid_results_path = HpoExperiment.get_output_path(
        base_output_path=base_results_path,
        algorithm_type=HpoAlgorithmType.GRID,
        tune_dataset=tune_dataset,
    )
    logger.info(f"Loading grid results from '{grid_results_path}'..")
    grid_results_path = grid_results_path / "tuning"
    return HpoResults.from_csv(grid_results_path)


def read_all_seeds_test_results(results_path):
    print(f"Reading all seeds test results from '{results_path}'..")
    seed_result_dirs = [d.name for d in results_path.iterdir() if d.is_dir()]
    result = []
    for seed_result_dir in seed_result_dirs:
        if seed_result_dir.startswith("seed_"):
            test_results_path = results_path / seed_result_dir / "test"
            seed_test_results = TestResults.from_csv(
                test_results_path, file_name="test_results.csv"
            )
            seed_test_results._results_summary["iteration_index"] = range(
                1, len(seed_test_results._results_summary) + 1
            )
            result.append(seed_test_results)
    if not result:
        raise RuntimeError(f"No seed results found in '{results_path}'.")
    result = TestResults.concat(result)
    return result


def aggregate_multi_seed_results(
    base_results_path: Path,
    algorithm_type: HpoAlgorithmType,
    analyzed_metric: MetricDefinition,
    tune_dataset: DatasetID,
    test_dataset: DatasetID,
    algorithm_display_name: str,
    sampling_setup: str,
    dataset_name: DatasetName,
) -> list[dict]:
    """Load and aggregate multi-seed test results for a single algorithm.

    Args:
        base_results_path: Base path for experiment results
        algorithm_type: Type of HPO algorithm
        analyzed_metric: MetricDefinition with metric names
        tune_dataset: DatasetID for tuning dataset
        test_dataset: DatasetID for test dataset
        algorithm_display_name: Display name of the algorithm
        sampling_setup: Sampling setup identifier
        dataset_name: Dataset name enum

    Returns:
        List of dictionaries containing metric results per iteration
    """
    # Get the path to test results
    tune_results_path = HpoExperiment.get_output_path(
        base_output_path=base_results_path,
        algorithm_type=algorithm_type,
        tune_dataset=tune_dataset,
        optimization_metric_id=analyzed_metric.name,
        test_dataset=test_dataset,
    )
    logger.info(f"Reading results from '{tune_results_path}'...")

    # Read the test file from the path using MultiSeedTestResults
    multi_seed_test_results = MultiSeedTestResults.from_csv(tune_results_path)

    results = multi_seed_test_results.aggregate_by_iteration(
        analyzed_metric_name=analyzed_metric.name,
        analyzed_metric_short_name=analyzed_metric.short_name,
    )

    # Add metadata to each result dictionary
    for result in results:
        result["algorithm"] = algorithm_display_name
        result["sampling_setup"] = sampling_setup
        result["dataset"] = dataset_name

    return results


def analyze_test_results(
    sampling_setups: list[SamplingSetupType],
    analyzed_metric: MetricDefinition,
    dataset_names: list[DatasetName],
    base_results_path: Path,
    analysis_path: Path,
):
    all_datasets_results = []
    metric_short_name = analyzed_metric.short_name

    algorithm_setups = get_algorithm_setups()
    for dataset_name in dataset_names:
        dataset_results = []
        for sampling_setup in sampling_setups:
            data_sampling_params = get_sampling_params(sampling_setup, dataset_name)

            # Create DatasetID for the tune dataset
            tune_dataset = DatasetID(
                dataset_name=dataset_name,
                split="Dev",
                sampling_params=data_sampling_params,
            )
            test_dataset = DatasetID(
                dataset_name=dataset_name,
                split="Test",
                sampling_params=data_sampling_params,
            )

            for algorithm_type in algorithm_setups:
                algorithm_display_name = algorithm_type.value

                # Aggregate multi-seed results for this algorithm
                algorithm_results = aggregate_multi_seed_results(
                    base_results_path=base_results_path,
                    algorithm_type=algorithm_type,
                    analyzed_metric=analyzed_metric,
                    tune_dataset=tune_dataset,
                    test_dataset=test_dataset,
                    algorithm_display_name=algorithm_display_name,
                    sampling_setup=sampling_setup,
                    dataset_name=dataset_name,
                )
                dataset_results.extend(algorithm_results)

        dataset_results = pd.DataFrame(dataset_results)
        dataset_test_analysis_path = (
            analysis_path / f"test_results_{metric_short_name}_{dataset_name}.csv"
        )
        # Ensure the directory exists before saving
        dataset_test_analysis_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_results.to_csv(dataset_test_analysis_path)
        logger.info(
            f"Results for '{metric_short_name}' and '{dataset_name}' written to '{dataset_test_analysis_path}'."
        )
        all_datasets_results.append(dataset_results)

        # algorithm_order = [algorithm_type.value for algorithm_type in algorithm_setups]
        # write_metric_results_for_paper(
        #    dataset_results, metric_short_name, algorithm_order, analysis_path
        # )

    all_datasets_results = pd.concat(all_datasets_results)
    return all_datasets_results


def write_metric_results_for_paper(
    metric_results: pd.DataFrame,
    metric_short_name: str,
    algorithm_order: list[str],
    analysis_path: Path,
):
    paper_metric_results = metric_results[
        [metric_short_name, "dataset", "algorithm", "sampling_setup"]
    ].copy()
    paper_metric_results["dataset"] = paper_metric_results["dataset"].apply(lambda v: v.value)

    paper_metric_results["algorithm"] = pd.Categorical(
        paper_metric_results["algorithm"], categories=algorithm_order, ordered=True
    )
    sampling_setup_order = ["sample", "full"]
    paper_metric_results["sampling_setup"] = pd.Categorical(
        paper_metric_results["sampling_setup"], categories=sampling_setup_order, ordered=True
    )
    paper_metric_results = paper_metric_results.sort_values(
        by=["dataset", "algorithm", "sampling_setup"], ascending=[True, True, True]
    )

    paper_metric_results = paper_metric_results.set_index(
        ["dataset", "algorithm", "sampling_setup"]
    )
    paper_metric_results = paper_metric_results.unstack(["algorithm", "sampling_setup"])
    paper_metric_results = paper_metric_results.applymap(lambda v: f"{v * 100:.1f}")

    paper_metrics_output_path = analysis_path / f"test_results_for_paper_{metric_short_name}.csv"
    # Ensure the directory exists before saving
    paper_metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    paper_metric_results.to_csv(paper_metrics_output_path)

    paper_metrics_latex = paper_metric_results.to_latex()
    print(f"------------\n{paper_metrics_latex}----------------")
    paper_metrics_latex_path = analysis_path / f"test_results_for_paper_{metric_short_name}.txt"
    with open(paper_metrics_latex_path, "w") as paper_metrics_latex_file:
        paper_metrics_latex_file.write(paper_metrics_latex)


def plot_performance_per_iteration(
    test_results: pd.DataFrame,
    analyzed_metric: MetricDefinition,
    analysis_path: Path,
    base_results_path: Path,
    with_sampling: bool,
):
    metric_short_name = analyzed_metric.short_name
    metric_display_name = analyzed_metric.display_name
    metric_internal_name = analyzed_metric.name
    metric_analysis_path = analysis_path / metric_short_name
    metric_analysis_path.mkdir(parents=True, exist_ok=True)

    performance_per_iteration = test_results[
        [metric_short_name, "dataset", "algorithm", "sampling_setup", "iteration_index"]
    ].copy()

    for single_dataset_fig in False, True:
        ax_index = 0
        axes = []
        fig = None
        results_per_dataset = performance_per_iteration.groupby("dataset")
        num_datasets = len(results_per_dataset)
        if not single_dataset_fig:
            # figsize=(10, 6) is good for 3 columns and 2 rows
            # num_datasets * 2.8
            height_in_inch = 14 if not with_sampling else 9  # for 5 datasets
            fig, axes = plt.subplots(num_datasets, 1, figsize=(6, height_in_inch))
            if num_datasets > 1:
                axes = axes.flat
            # for ax in axes:
            #    ax.set_aspect(20)
        for dataset_index, (dataset, dataset_results) in enumerate(results_per_dataset, start=1):
            ax: Axes
            if single_dataset_fig:
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
            else:
                if num_datasets > 1:
                    ax = axes[ax_index]
                else:
                    ax = cast(Axes, axes)
                ax_index += 1
            for algorithm_label, algorithm_results in dataset_results.groupby("algorithm"):
                by_sampling_setup = algorithm_results.groupby("sampling_setup")
                for sampling_setup, sampling_setup_results in by_sampling_setup:
                    by_iteration_results = sampling_setup_results[
                        ["iteration_index", metric_short_name]
                    ]
                    by_iteration_results = by_iteration_results.set_index("iteration_index")

                    label, color, linestyle, linewidth = get_plot_props(
                        algorithm_label, sampling_setup if with_sampling else None
                    )
                    ax.plot(
                        by_iteration_results.index,
                        by_iteration_results[metric_short_name],
                        label=label,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        color=color,
                    )

            title = str(dataset)
            ax.set_title(title)
            ax.set_xlabel("# Iterations")
            ax.set_ylabel(metric_display_name)
            iterations_range = range(1, 11)
            ax.set_xticks(iterations_range)
            ax.set_xlim(by_iteration_results.index[0], by_iteration_results.index[-1])

            # Plot the best possible test results from the test grid search results
            plot_test_grid = True
            if plot_test_grid:
                test_grid_results = get_grid_results(
                    dataset_name=dataset,
                    split="Test",
                    base_results_path=base_results_path,
                )

                best_test_result = test_grid_results._results_summary[
                    f"{metric_internal_name}_mean"
                ].max()
                ax.axhline(y=best_test_result, color="black", linewidth=1.5, linestyle="--")

            ymin, ymax = ax.get_ylim()
            ax.vlines(
                x=iterations_range,
                ymin=ymin,
                ymax=ymax,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.3,
            )

            if single_dataset_fig:
                ax.legend(loc="lower right", fontsize=6, ncol=2)
                test_results_path = metric_analysis_path / f"test_results_{dataset}.png"
                plt.savefig(test_results_path, dpi=300)
                logger.info(f"test results per iteration written to '{test_results_path}'.")
                test_results_path = metric_analysis_path / f"test_results_{dataset}.pdf"
                plt.savefig(test_results_path, dpi=300)
            last_dataset = dataset_index == num_datasets
            if not single_dataset_fig and last_dataset:
                # -0.35 for 5 datasets
                ncol = 3 if with_sampling else 2
                bbox_to_anchor_height = -0.35 if not with_sampling else -0.15
                ax.legend(
                    loc="upper center", bbox_to_anchor=(0.5, bbox_to_anchor_height), ncol=ncol
                )  # fontsize=6,

        if not single_dataset_fig:
            if not with_sampling:
                fig.subplots_adjust(hspace=0.6)  # for 5 datasets
            else:
                fig.subplots_adjust(hspace=0.36, bottom=0.16, top=0.96)
            # plt.show()
            # fig.subplots_adjust(hspace=0.4)  # for 2 datasets
            # ax = axes[-1]
            # ax.axis("off")
            sampling_setup_name = "sample" if with_sampling else "full"
            test_results_path = (
                analysis_path
                / ".."
                / f"test-results_data-{sampling_setup_name}_{metric_short_name}.png"
            )
            # Ensure the directory exists before saving
            test_results_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(test_results_path, dpi=600)
            logger.info(f"test results per iteration written to '{test_results_path}'.")
            test_results_path = (
                analysis_path
                / ".."
                / f"test-results_data-{sampling_setup_name}_{metric_short_name}.pdf"
            )
            plt.savefig(test_results_path, dpi=600)


def get_plot_props(algorithm_label, sampling_setup):
    label = algorithm_label
    if sampling_setup:
        sampling_setup = "Full" if sampling_setup == "full" else "Sample"
        label += f"/{sampling_setup}"

    linestyle = "-"
    linewidth = 2
    color = "yellow"  # Default color (lowercase for matplotlib)

    # Check algorithm type using HpoAlgorithmType enum values
    label_lower = label.lower()
    if HpoAlgorithmType.GRID.value in label_lower:
        linewidth = 1.5
        color = "red"
    elif HpoAlgorithmType.RANDOM.value in label_lower:
        color = "blue"
    elif HpoAlgorithmType.GREEDY_M.value in label_lower:
        color = "purple"
    elif HpoAlgorithmType.GREEDY_R.value in label_lower:
        color = "pink"

    # Adjust linestyle for sampling
    if "sample" in label_lower:
        linestyle = ":"

    return label, color, linestyle, linewidth


def analyze_test_results_main(
    with_sampling: bool, analyzed_metric: MetricDefinition, base_results_path: Path
):
    if not with_sampling:
        dataset_names = [
            DatasetName.WatsonxQA,
            DatasetName.MiniWiki,
            DatasetName.AIArxiv,
            DatasetName.ClapNQ,
            DatasetName.BioASQ,
        ]
    else:
        dataset_names = [
            DatasetName.ClapNQ,
            DatasetName.BioASQ,
        ]

    sampling_setups: list[SamplingSetupType] = ["full"]
    if with_sampling:
        sampling_setups.append("sample")
    analysis_path = (
        base_results_path / f"analysis/test_results/{'sample' if with_sampling else 'full'}"
    )
    test_results = analyze_test_results(
        sampling_setups=sampling_setups,
        analyzed_metric=analyzed_metric,
        dataset_names=dataset_names,
        base_results_path=base_results_path,
        analysis_path=analysis_path,
    )

    plot_performance_per_iteration(
        test_results, analyzed_metric, analysis_path, base_results_path, with_sampling
    )


def run_analysis(base_results_path: Path | None = None):
    """Run analysis on test results for all metrics.

    Args:
        base_results_path: Base path for experiment results. If None, uses default path.
    """
    if base_results_path is None:
        base_results_path = Path(__file__).parent.parent.parent.parent / "experiments_output"

    analyzed_metrics = [
        LLMAAJ_AC,
        LEXICAL_AC,
        LEXICAL_FF,
    ]
    for with_sampling in [False]:
        for analyzed_metric in analyzed_metrics:
            analyze_test_results_main(with_sampling, analyzed_metric, base_results_path)


if __name__ == "__main__":
    init_logger()
    run_analysis()
