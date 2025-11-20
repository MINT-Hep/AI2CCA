import os
from typing import List
import numpy as np
import pandas as pd

EPS = 1e-12


def _entropy_from_probs(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Compute binary entropy for each sample: -sum p log p."""
    p0c = np.clip(p0, EPS, 1.0 - EPS)
    p1c = np.clip(p1, EPS, 1.0 - EPS)
    return -(p0c * np.log(p0c) + p1c * np.log(p1c))


def compute_predictive_entropy(
    csv_paths: List[str],
    output_path: str = "ensemble_predictive_entropy.xlsx",
) -> pd.DataFrame:
    """
    Compute predictive entropy (H(E[p])) across multiple model CSVs and
    derive final_result from mean probabilities.

    Args:
        csv_paths: List of CSV file paths
                   (each must contain columns: slide_id, Y, p_0, p_1)
        output_path: Path to save the result (.csv or .xlsx)

    Returns:
        DataFrame with columns:
            slide_id, label, final_result, p0_mean, p1_mean, predictive_entropy
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        required = {"slide_id", "Y", "p_0", "p_1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        # keep label as Y here; we will aggregate later
        dfs.append(df[["slide_id", "Y", "p_0", "p_1"]])

    if not dfs:
        raise ValueError("No valid CSV files provided.")

    combined = pd.concat(dfs, ignore_index=True)

    # Mean probabilities per slide across models
    prob_agg = combined.groupby("slide_id").agg(
        p0_mean=("p_0", "mean"),
        p1_mean=("p_1", "mean"),
    ).reset_index()

    # Predictive entropy: H(E[p])
    prob_agg["predictive_entropy"] = _entropy_from_probs(
        prob_agg["p0_mean"].to_numpy(),
        prob_agg["p1_mean"].to_numpy(),
    )

    # Final result from mean probabilities (binary: 0/1)
    # Here we use >= so ties go to class 1
    prob_agg["final_result"] = (prob_agg["p1_mean"] >= prob_agg["p0_mean"]).astype(int)

    # Get label per slide (assume all models share same label; take first)
    label_agg = (
        combined.groupby("slide_id")["Y"]
        .first()  # or .mode().iloc[0] if you want majority vote
        .rename("label")
        .reset_index()
    )

    # Merge label + stats
    out = (
        prob_agg.merge(label_agg, on="slide_id", how="left")
        .loc[:, ["slide_id", "label", "final_result", "p0_mean", "p1_mean", "predictive_entropy"]]
        .sort_values("slide_id")
        .reset_index(drop=True)
    )

    # Save
    if output_path.lower().endswith(".xlsx"):
        out.to_excel(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    import glob

    # Example usage
    data_dir = "./results_example"
    csv_list = sorted(glob.glob(os.path.join(data_dir, "fold_*.csv")))
    if not csv_list:
        raise SystemExit("No CSV files found. Please check your path or filename pattern.")
    df = compute_predictive_entropy(
        csv_list, output_path=os.path.join(data_dir, "ensemble_predictive_entropy.xlsx")
    )
    print("Predictive entropy saved.")
    print(df.head())
