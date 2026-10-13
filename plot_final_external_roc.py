"""Plot three frozen external ROC curves without training or cutoff selection."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def main():
    default_predictions = (
        Path(__file__).resolve().parent
        / "nested_grouped_ml_final/48_three_stage_final_comparison"
        / "external_validation/chinese_external_predictions.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=default_predictions)
    parser.add_argument("--output", type=Path, help="Output filename stem; saves PNG, PDF and SVG.")
    parser.add_argument("--show", action="store_true", help="Show the figure after saving.")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions, dtype={"id": str}, float_precision="round_trip")
    models = [
        ("clinical_only", "Clinical-only", "#707880"),
        ("microbiome_only", "Microbiome-only", "#218779"),
        ("combined", "Combined", "#E56B50"),
    ]
    if set(df["model"]) != {name for name, _, _ in models}:
        raise ValueError("Expected exactly Clinical-only, Microbiome-only and Combined.")
    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    reference = None
    for model, label, color in models:
        part = df.loc[df["model"] == model].sort_values("id")
        if part["id"].isna().any() or part["id"].duplicated().any():
            raise ValueError(f"Missing or duplicate IDs: {model}")
        if set(part["true_response"]) != {0, 1}:
            raise ValueError(f"Outcome must contain both 0 and 1: {model}")
        current = part.set_index("id")["true_response"]
        if reference is not None and not current.equals(reference):
            raise ValueError("The three models must use identical patients and labels.")
        reference = current
        y = part["true_response"].to_numpy()
        p = part["probability"].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid predicted probabilities: {model}")
        fpr, tpr, _ = roc_curve(y, p, pos_label=1)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{label}: AUROC = {auc:.3f}")
        print(f"{label}: N={len(y)}, responders={int(y.sum())}, AUROC={auc:.6f}")
    ax.plot([0, 1], [0, 1], "--", color="#A5A9AE", lw=1.2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)",
           title=f"Chinese external validation (N = {len(reference)})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if args.output:
        outputs = [Path(f"{args.output}.{extension}") for extension in ("png", "pdf", "svg")]
        if any(path.exists() for path in outputs):
            raise FileExistsError("Output exists; choose a new filename stem.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=300)
    if args.show or not args.output:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
