"""Draw frozen external confusion matrices with response as the positive class."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Wedge
from sklearn.metrics import confusion_matrix


def main():
    root = Path(__file__).resolve().parent
    archive = root / "nested_grouped_ml_final/48_three_stage_final_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=archive)
    parser.add_argument("--output", type=Path, default=root / "confusion_matrix_figures")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve existing files.")
    predictions = pd.read_csv(args.archive / "external_validation"
                              / "chinese_external_predictions.csv",
                              dtype={"id": str}, float_precision="round_trip")
    registry = json.loads((args.archive / "combined_oof_threshold_supplement/results"
                           / "three_model_public_oof_thresholds.json").read_text())
    reference_metrics = pd.read_csv(args.archive / "combined_oof_threshold_supplement/results"
                                    / "three_model_external_performance.csv")
    metadata = pd.read_csv(args.archive / "input_snapshot"
                           / "yyyy_external_0928_pwy_matched_integrated_species_meta.tsv",
                           sep="\t", dtype={"id": str})
    if metadata["id"].isna().any() or metadata["id"].duplicated().any():
        raise ValueError("Invalid archived metadata IDs.")
    reference_labels = metadata.set_index("id")["group"].sort_index()
    models = [("clinical_only", "Clinical-only"), ("microbiome_only", "Microbiome-only"),
              ("combined", "Combined")]
    records, matrices = [], {}
    for model, _ in models:
        part = predictions.loc[predictions["model"] == model].sort_values("id")
        if part["id"].isna().any() or part["id"].duplicated().any():
            raise ValueError(f"Invalid prediction IDs: {model}")
        if not part.set_index("id")["true_response"].equals(reference_labels):
            raise ValueError(f"Prediction labels do not match archived metadata: {model}")
        if set(part["true_response"]) != {0, 1}:
            raise ValueError("Both response classes must be represented.")
        if registry[model]["source"] != "public_oof":
            raise ValueError("Only locked public OOF cutoffs are permitted.")
        p = part["probability"].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid prediction probabilities.")
        cutoff = float(registry[model]["threshold"])
        cm = confusion_matrix(part["true_response"], p >= cutoff, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        expected = reference_metrics.loc[reference_metrics["model"] == model].iloc[0]
        for name, value in (("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp)):
            if value != int(expected[name]):
                raise ValueError(f"Archived confusion-count mismatch: {model}/{name}")
        matrices[model] = cm
        records.append({"model": model, "N": len(part), "cutoff": cutoff,
                        "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    args.output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    vmax = max(int(cm.max()) for cm in matrices.values())
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("rose_counts", ["#FBECEF", "#D86D7D"])
    for model, label in models:
        cm = matrices[model]
        cutoff = float(registry[model]["threshold"])
        fig = plt.figure(figsize=(5.6, 4.7), facecolor="white")
        ax = fig.add_axes([0.23, 0.17, 0.49, 0.60])
        sectors = [(0, 0, 90, 180, "TN", -0.92, 0.92),
                   (0, 1, 0, 90, "FP", 0.92, 0.92),
                   (1, 0, 180, 270, "FN", -0.92, -0.92),
                   (1, 1, 270, 360, "TP", 0.92, -0.92)]
        for row, col, start, end, name, tx, ty in sectors:
            count = int(cm[row, col])
            radius = 0.87 * np.sqrt(count / vmax)
            ax.add_patch(Wedge((0, 0), radius, start, end,
                               facecolor=cmap(norm(count)), edgecolor="none"))
            percent = 100 * count / cm[row].sum()
            ax.text(tx, ty, f"{name}\n({percent:.1f}%)", fontsize=9, fontweight="bold",
                    ha="left" if col == 0 else "right",
                    va="top" if row == 0 else "bottom")
            ax.text(-0.19 if col == 0 else 0.19, 0.19 if row == 0 else -0.19,
                    str(count), ha="center", va="center", fontweight="bold", fontsize=12)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.axvline(0, color="#777777", lw=0.8)
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#333333", lw=1))
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
        ax.set_xticks([-0.5, 0.5], ["Non-response", "Response"], fontweight="bold")
        ax.set_yticks([0.5, -0.5], ["Non-response", "Response"], rotation=90,
                      va="center", fontweight="bold")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0, pad=5)
        ax.set_xlabel("Predicted label", labelpad=9, fontweight="bold")
        ax.set_ylabel("True label", labelpad=13, fontweight="bold")
        ax.spines[:].set_visible(False)
        cax = fig.add_axes([0.80, 0.33, 0.03, 0.29])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cax.set_title("Count value", fontsize=9, loc="left", pad=9)
        cax.tick_params(length=0, labelsize=9)
        cax.spines[:].set_visible(False)
        fig.text(0.5, 0.955, label, ha="center", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.09, f"Chinese external validation: N = {int(cm.sum())}",
                 ha="center", fontsize=9)
        fig.text(0.5, 0.05, f"Response = 1 | Locked public OOF cutoff = {cutoff:.6f}",
                 ha="center", fontsize=8, color="#555555")
        for extension in ("pdf", "png", "svg"):
            fig.savefig(args.output / f"{model}_confusion_matrix.{extension}", dpi=300)
        plt.close(fig)
        print(label, cm.tolist())
    pd.DataFrame(records).to_csv(args.output / "confusion_matrix_counts.csv", index=False)
    (args.output / "plot_notes.json").write_text(json.dumps({
        "positive_label": "Response (1)", "negative_label": "Non-response (0)",
        "rows": ["Non-response", "Response"], "columns": ["Non-response", "Response"],
        "percentages": "Row-normalized: denominator is the corresponding true class total.",
        "sector_area": "Proportional to count, with common scale across three models.",
        "models_retrained": False, "thresholds_changed": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
