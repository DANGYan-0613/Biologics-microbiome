"""Plot archived ExtraTrees impurity importance, without reading patient data."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


COLORS = {"Species": "#218779", "PWY": "#E56B50", "Clinical": "#707880"}


def display_name(feature):
    if feature.startswith("species__species__s__"):
        return feature.removeprefix("species__species__s__").replace("_", " ")
    if feature.startswith("pwy__pwy__"):
        return feature.removeprefix("pwy__pwy__")
    return {
        "clinical__subtype_2": "Subtype: CD (vs UC)",
        "clinical__biologic_VDZ": "Biologic: VDZ (vs Anti-TNF)",
        "clinical__biologic_UST": "Biologic: UST (vs Anti-TNF)",
    }.get(feature, feature)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "nested_grouped_ml_final"
                        / "48_three_stage_final_comparison")
    parser.add_argument("--output", type=Path, default=root / "feature_importance_figures")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing outputs are preserved.")
    selected = pd.read_csv(args.source / "features"
                           / "microbiome_only_public175_final_selected_features.csv")
    if selected["feature"].duplicated().any():
        raise ValueError("Duplicate selected microbial features.")
    selected_microbes = set(selected["feature"])
    tables, audit = {}, {}
    for model in ("microbiome_only", "combined"):
        path = args.source / "feature_importance" / f"{model}_feature_importance.csv"
        table = pd.read_csv(path, float_precision="round_trip")
        if table["feature"].isna().any() or table["feature"].duplicated().any():
            raise ValueError(f"Missing or duplicate features: {model}")
        if table["feature_type"].isna().any() or not set(table["feature_type"]).issubset(COLORS):
            raise ValueError(f"Unknown feature types: {model}")
        values = table["model_importance"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid importance: {model}")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError(f"Importance must sum to one: {model}")
        if not np.allclose(table["importance_percent"], values * 100, atol=1e-12, rtol=0):
            raise ValueError(f"Importance percentage mismatch: {model}")
        microbial = table.loc[table["feature_type"].isin(["Species", "PWY"]), "feature"]
        if set(microbial) != selected_microbes:
            raise ValueError(f"Importance does not match the frozen microbial signature: {model}")
        table = table.sort_values("model_importance", ascending=False, kind="stable").reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        table["display_name"] = table["feature"].map(display_name)
        tables[model] = table
        audit[model] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "number_features": len(table),
            "counts_by_type": table["feature_type"].value_counts().to_dict(),
            "importance_sum": float(values.sum()),
        }
    args.output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    for model, table in tables.items():
        label = "Microbiome-only" if model == "microbiome_only" else "Combined"
        fig, ax = plt.subplots(figsize=(11.5, max(9, len(table) * 0.4 + 1.5)))
        fig.subplots_adjust(left=0.57, right=0.96, bottom=0.08, top=0.88)
        y = np.arange(len(table))
        values = table["importance_percent"].to_numpy()
        ax.barh(y, values, height=0.66,
                color=table["feature_type"].map(COLORS), edgecolor="none")
        labels = [textwrap.fill(f"{rank:02d}  {name}", width=57, break_long_words=False,
                                break_on_hyphens=False)
                  for rank, name in zip(table["rank"], table["display_name"])]
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.invert_yaxis()
        ax.set_xlim(0, values.max() * 1.18)
        ax.set_xlabel("ExtraTrees impurity-based importance (%)", labelpad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E4E6E8", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B7BCC1")
        for row, value in enumerate(values):
            ax.text(value + values.max() * 0.018, row, f"{value:.2f}%",
                    va="center", fontsize=8.5)
        counts = audit[model]["counts_by_type"]
        fig.text(0.055, 0.974, f"{label}: ranked feature importance", fontsize=16, va="top")
        fig.text(0.055, 0.946, "Final public-trained model (N = 175) | "
                 + " + ".join(f"{counts[k]} {k}" for k in COLORS if k in counts),
                 fontsize=10, color="#565D64", va="top")
        fig.legend(handles=[Patch(facecolor=color, label=kind)
                            for kind, color in COLORS.items() if kind in counts],
                   loc="upper left", bbox_to_anchor=(0.049, 0.923),
                   ncol=len(counts), frameon=False)
        stem = args.output / f"{model}_ranked_feature_importance"
        for extension in ("pdf", "png", "svg"):
            fig.savefig(f"{stem}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)
        table.to_csv(f"{stem}.csv", index=False)
        print(f"{label}: {len(table)} features; saved {stem}.pdf")
    audit["method"] = "Frozen fitted-model mean decrease in impurity; not PLS VIP."
    audit["models_refitted"] = False
    audit["external_data_read"] = False
    (args.output / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
