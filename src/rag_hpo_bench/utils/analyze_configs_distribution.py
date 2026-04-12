import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame

INPUT_FOLDER = Path(
    "/Users/matano/Library/CloudStorage/Box-Box/matan/ongoing/2025 autorag AAAI workshop submission/paper results"
)
# 🔁 Change to your parent folder path# === Configuration ===
PARAMETER_PATHS = {
    "evaluation_results|pattern_name": "pattern_name",
    "evaluation_results|indexing_params|chunking_size": "chunking_size",
    "evaluation_results|indexing_params|chunking_overlap": "chunking_overlap",
    "evaluation_results|indexing_params|embedding_model": "embedding_model",
    "evaluation_results|indexing_params|dataset_id": "dataset_id",
    "evaluation_results|inference_params|number_of_retrieved_chunks": "top-k",
    "evaluation_results|inference_params|inference_model_id": "generator",
    "evaluation_results|scores|scores|answer_correctness|mean": "Lexical-AC",
    "evaluation_results|scores|scores|faithfulness|mean": "Lexical-FF",
    "evaluation_results|scores|scores|context_correctness|mean": "context_correctness",
    "evaluation_results|scores|scores|ragas.answer_correctness.gpt-4o-mini-2024-07-18|mean": "LLMaaJ-AC",
}  # 🔁 Dot-separated parameter paths to extract
metric_names_mapping = {
    "answer_correctness": "Lexical-AC",
    "faithfulness": "Lexical-FF",
    "context_correctness": "context_correctness",
    "ragas.answer_correctness.gpt-4o-mini-2024-07-18": "LLMaaJ-AC",
}
METRIC_COLUMNS = ["Lexical-AC", "Lexical-FF", "LLMaaJ-AC"]
OUTPUT_FILE = Path("output/paper_results.csv")  # 🔁 Set to None to skip saving

# === Helper Functions ===


def get_nested_value(data, dotted_path: str):
    """
    Resolve a dotted path strictly through dictionaries.
    - If any level is a list, raise an exception (lists are not supported).
    - If a key is missing, return None (treat as absent, not an error).
    """
    keys = dotted_path.split("|")
    cur = data
    traversed = []

    for key in keys:
        traversed.append(key)
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
            else:
                # Missing key is acceptable -> value is None
                return None
        elif isinstance(cur, list):
            path_so_far = ".".join(traversed[:-1])
            raise RuntimeError(
                f"Lists are not supported in parameter paths. "
                f"Encountered a list at '{path_so_far}' while resolving '{dotted_path}'."
            )
        else:
            # Hit a scalar/None before finishing the path
            return None

    return cur


def extract_json_values(file_path: Path, param_paths: dict[str, str]):
    """Extract specified parameters from a JSON file, return a dict for one row."""
    row = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "parent_folder": file_path.parent.name,
    }
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        # Still return file metadata; parameter columns will be NaN
        for param_name in param_paths.values():
            row.setdefault(param_name, None)
        return row

    for param_path, param_name in param_paths.items():
        row[param_name] = get_nested_value(data, param_path)
    return row


def plot_score_histogram(
    df: pd.DataFrame,
    dataset_ids: dict[str, str],
    metric_column: str,
    output_pdf_path: str,
):
    required_cols = {"dataset_id", metric_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    metric_by_dataset_id: dict[str, pd.Series] = {}
    for dataset_id in dataset_ids.keys():
        df_filt = df[df["dataset_id"] == dataset_id].copy()

        metric_series = pd.to_numeric(df_filt[metric_column], errors="coerce").dropna()
        if metric_series.empty:
            raise ValueError(
                f"All values in column {metric_column!r} are non-numeric or NaN "
                f"after filtering for dataset_id == {dataset_id!r}."
            )

        s_min, s_max = metric_series.min(), metric_series.max()
        if s_max == s_min:
            s_norm = pd.Series(np.zeros(len(metric_series)), index=metric_series.index)
        else:
            s_norm = (metric_series - s_min) / (s_max - s_min)
        metric_by_dataset_id[dataset_id] = s_norm

    if not metric_by_dataset_id:
        raise ValueError("No valid data to plot.")

    colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get(
            "color",
            [
                "#4C78A8",
                "#F58518",
                "#54A24B",
                "#E45756",
                "#72B7B2",
                "#FF9DA6",
                "#B279A2",
                "#9D755D",
                "#BAB0AC",
            ],
        )
    )

    n_datasets = len(metric_by_dataset_id)
    bins = np.linspace(0.0, 1.0, 11)  # 10 bins over [0,1]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    group_width = min(0.9, n_datasets * (bin_width * 0.8) / n_datasets)
    bar_width = group_width / n_datasets
    offsets = (np.arange(n_datasets) - (n_datasets - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, (dsid, s_norm) in enumerate(metric_by_dataset_id.items()):
        counts, _ = np.histogram(s_norm, bins=bins)
        print(f"{dsid}: {sum(counts)}")
        perc = counts / counts.sum() * 100.0 if counts.sum() > 0 else counts.astype(float)
        ax.bar(
            bin_centers + offsets[i],
            perc,
            width=bar_width * 0.95,
            color=colors[i % len(colors)],
            edgecolor="white",
            label=str(dataset_ids[dsid]),
            align="center",
        )

    fontsize = 18

    # Configure x-axis ticks to show bin intervals
    num_bins = len(bins)
    tick_labels = [
        f"[{bins[i]:.1f}–{bins[i + 1]:.1f}" + ("]" if (i == (num_bins - 2)) else ")")
        for i in range(num_bins - 1)
    ]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(tick_labels, rotation=0)
    # ax.set_xticklabels(tick_labels, rotation=30, ha="right")  # rotate for better readability

    # Reduce font size for tick labels
    ax.tick_params(axis="x", labelsize=10)  # adjust size as needed (e.g., 10–12)

    # Set limits to cover the full [0,1] range with slight padding
    ax.set_xlim(bins[0] - bin_width * 0.05, bins[-1] + bin_width * 0.05)

    ax.set_xlabel(f"{metric_column}", fontsize=fontsize)
    ax.set_ylabel("% of RAG configurations", fontsize=fontsize)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Dataset", ncol=1, fontsize=12, title_fontsize=12, loc="upper left")
    ax.tick_params(axis="y", which="major", labelsize=14)

    plt.tight_layout()

    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)
        print(f"Saved plot to '{output_pdf_path}'.")


def create_published_config_files(
    df,
    output_path: Path,
    overwrite_output: bool = False,
    with_json: bool = False,
    zip_output: bool = False,
):
    destination_folders = set()
    for config_id, dataset, split, source_file_path in zip(
        df["config_id"], df["Dataset"], df["Split"], df["file_path"], strict=False
    ):
        destination_path = output_path / dataset / split / Path(source_file_path).name
        destination_folders.add(destination_path.parent)
        destination_path.parent.mkdir(exist_ok=True, parents=True)
        if with_json:
            shutil.copy(source_file_path, destination_path)
        destination_csv_path = destination_path.parent / f"RagConfiguration{config_id}.csv"
        if (not destination_csv_path.exists()) or overwrite_output:
            pattern_results_to_csv(json_path=source_file_path, csv_path=destination_csv_path)

    if zip_output:
        zip_folders(
            destination_folders,
            expected_file_count=162,
            zip_output_dir=output_path / "zip",
        )


def zip_folders(destination_folders: set[Path], expected_file_count: int, zip_output_dir: Path):
    """
    For each folder in destination_folders:
      - check that it has exactly expected_file_count files
      - zip all files inside it into zip_output_dir / <folder_name>.zip
    """

    destination_folders = [Path(p) for p in destination_folders]
    zip_output_dir = Path(zip_output_dir)
    zip_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"zipping {len(destination_folders)} folders ..")
    for folder in destination_folders:
        if not folder.exists() or not folder.is_dir():
            print(f"❌ Folder does not exist or is not a directory: {folder}")
            continue

        # List *files* only (not directories)
        files = [f for f in folder.iterdir() if f.is_file()]

        # Validate file count
        if len(files) != expected_file_count:
            print(
                f"❌ Folder {folder} has {len(files)} files "
                f"(expected {expected_file_count}). Skipped."
            )
            continue

        # Create output zip path
        zip_path = zip_output_dir / f"{folder.name}.zip"

        # Zip all files in the folder
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, arcname=f.name)

        print(f"✅ Zipped {len(files)} files from {folder} → {zip_path}")


def pattern_results_to_csv(json_path, csv_path):
    """
    Load the JSON evaluation file, convert evaluation_data + question_scores
    into a pandas DataFrame, preserving list fields for ground_truths and contexts.
    """

    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    eval_data = data["evaluation_data"]
    question_scores = data["evaluation_results"]["scores"]["question_scores"]

    # All scoring categories (answer_correctness, faithfulness, etc.)
    score_categories = list(question_scores.keys())

    rows = []

    # Build rows
    for item in eval_data:
        qid = item["question_id"]

        row = {
            "question_id": qid,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "ground_truths": item.get("ground_truths", []),  # <-- keep list
            "contexts": item.get("contexts", []),  # <-- keep list
        }

        # Add per-question scores
        for score_cat in score_categories:
            row[score_cat] = question_scores[score_cat].get(qid, None)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    df = df.rename(columns=metric_names_mapping)
    df = df.drop(
        columns=[
            "metrics.llm_as_judge.binary.llama_3_1_70b_instruct_wml_answer_correctness_q_a_gt_loose_logprobs[inference_model=engines.classification.llama_3_1_405b_instruct_fp8_ibm_genai]",
            "ragas_answer_correctness_gpt-4o-mini-2024-07-18",
        ],
        errors="ignore",
    )

    # Write CSV (pandas writes Python lists as literal list strings)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # print(f"CSV saved to: {csv_path}")
    return df


def create_published_results(df: DataFrame, output_path: Path):
    df = df.copy()
    df[["Dataset", "Split"]] = df["dataset_id"].map(dataset_id_to_dataset_mapping).apply(pd.Series)
    print(f"# Results before dataset filtering: {len(df)}")
    df = df.dropna(subset=["Dataset"])
    print(f"# Results after dataset filtering: {len(df)}")

    df = add_config_id(df)

    create_published_config_files(df, output_path.parent)
    df = df[
        [
            "Dataset",
            "Split",
            "config_id",
            "chunking_size",
            "chunking_overlap",
            "embedding_model",
            "top-k",
            "generator",
            "context_correctness",
            "LLMaaJ-AC",
            "Lexical-AC",
            "Lexical-FF",
        ]
    ]
    df["chunking_overlap"] = df["chunking_overlap"] / df["chunking_size"]
    col_map = {
        "config_id": "Configuration ID",
        "chunking_size": "Chunk Size",
        "chunking_overlap": "Chunk Overlap",
        "embedding_model": "Embedding Model",
        "top-k": "Top-K",
        "generator": "Generative Model",
        "context_correctness": "Context Correctness",
    }
    df = df.rename(columns=col_map)

    df = df.sort_values(["Dataset", "Split", "Configuration ID"], ascending=[True, True, True])

    df.to_csv(output_path, index=False)


def add_config_id(df: DataFrame) -> DataFrame:
    param_cols = [
        "chunking_size",
        "chunking_overlap",
        "embedding_model",
        "top-k",
        "generator",
    ]

    # 1. Extract all unique parameter combinations
    unique_combos = (
        df[param_cols]
        .drop_duplicates()
        .sort_values(param_cols)  # define ordering explicitly
        .reset_index(drop=True)
    )

    # 2. Assign a consecutive integer ID to each unique combination
    unique_combos["config_id"] = range(len(unique_combos))

    print(f"Found {len(unique_combos)} unique RAG patterns.")

    # 3. Merge back into the original DataFrame
    df = df.merge(unique_combos, on=param_cols, how="left")
    return df


def main(allow_overwrite: bool):
    if not INPUT_FOLDER.is_dir():
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a directory: {INPUT_FOLDER}")

    rows = []

    if not OUTPUT_FILE.exists() or allow_overwrite:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Iterate over immediate subfolders of INPUT_FOLDER
        for sub in sorted(p for p in INPUT_FOLDER.iterdir() if p.is_dir()):
            # For each subfolder, process JSON files directly contained (non-recursive)
            json_files = sorted(sub.glob("*.json"))
            num_files = len(json_files)
            print(f"Found {num_files} json files in {sub}")
            for file_index, fp in enumerate(json_files, start=1):
                if file_index % 100 == 0:
                    print(f"Reading file {file_index}/{num_files}..")
                rows.append(extract_json_values(fp, PARAMETER_PATHS))

        # Build DataFrame
        df = pd.DataFrame(rows)

        # Ensure stable columns and presence even if empty
        base_cols = ["parent_folder", "file_name", "file_path"]
        for param_name in PARAMETER_PATHS.values():
            if param_name not in df.columns:
                df[param_name] = None

        cols = base_cols + list(PARAMETER_PATHS.values())
        if "error" in df.columns:
            cols += ["error"]

        # Reorder and print preview
        df = df[cols]
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(df.head())

        # Save if requested
        if OUTPUT_FILE:
            out = Path(OUTPUT_FILE)
            df.to_csv(out, index=False)
            print(f"✅ Saved output to: {out.resolve()}")
            published_results_file = out.with_name("rag_configurations_summary.csv")
            print(f"✅ Saving published results to: {published_results_file.resolve()}..")
            create_published_results(df, published_results_file)
    else:
        print(f"Loading existing results from {OUTPUT_FILE}..")
        df = pandas.read_csv(OUTPUT_FILE)

    for metric_column in METRIC_COLUMNS:
        plot_score_histogram(
            df,
            dataset_ids={
                "name-clap_nq_split-train_q-1000_seed-43": "ClapNQ",
                "name-bioasq_split-train_q-1000_seed-43": "BioASQ",
                "name-ai_arxiv_split-train": "AIArxiv",
                "name-mini_wikipedia_split-train": "MiniWiki",
                "name-watson_x_documents_split-train": "WatsonxQA",
            },
            metric_column=metric_column,
            output_pdf_path=f"configuration_distribution_{metric_column}.pdf",
        )


dataset_id_to_dataset_mapping = {
    # ai_arxiv
    "name-ai_arxiv_split-test": ("AIArxiv", "Test"),
    "name-ai_arxiv_split-train": ("AIArxiv", "Dev"),
    "name-ai_arxiv_split-train_q-40_docs-factor-9_seed-43": None,
    # bioasq
    "name-bioasq_split-train_q-100_docs-factor-9_seed-43": ("BioASQ", "Dev-Sampled"),
    "name-bioasq_split-train_q-1000_seed-43": ("BioASQ", "Dev"),
    "name-bioasq_split-test_q-150_seed-43": ("BioASQ", "Test"),
    "name-bioasq_split-test": None,
    "name-bioasq_split-train": None,
    # clap_nq
    "name-clap_nq_split-test_q-150_seed-43": ("ClapNQ", "Test"),
    "name-clap_nq_split-train_q-100_docs-factor-9_seed-43": ("ClapNQ", "Dev-Sampled"),
    "name-clap_nq_split-train_q-1000_seed-43": ("ClapNQ", "Dev"),
    "name-clap_nq_split-test": None,
    # mini_wikipedia
    "name-mini_wikipedia_split-train_q-66_seed-43": None,
    "name-mini_wikipedia_split-test_q-150_seed-43": ("MiniWiki", "Test"),
    "name-mini_wikipedia_split-train": ("MiniWiki", "Dev"),
    "name-mini_wikipedia_split-test": None,
    # watson_x_documents → WatsonxQA
    "name-watson_x_documents_split-train_q-40_docs-factor-9_seed-43": None,
    "name-watson_x_documents_split-test": ("WatsonxQA", "Test"),
    "name-watson_x_documents_split-train": ("WatsonxQA", "Dev"),
}


if __name__ == "__main__":
    main(allow_overwrite=True)
