"""Evaluation of frozen predictions only; no classifier or calibrator updates."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

LABELS = ("clinical_only", "microbiome_only", "combined")
BOOTSTRAPS = 10000


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z, p = float(norm.ppf(.975)), k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (0. if k == 0 else max(0., center - half), 1. if k == n else min(1., center + half))


def classification_intervals(tp, fp, tn, fn):
    counts = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp),
              "accuracy": (tp + tn, tp + fp + tn + fn), "PPV": (tp, tp + fp), "NPV": (tn, tn + fn)}
    estimates = {name: k / n if n else np.nan for name, (k, n) in counts.items()}
    intervals = {name: wilson(k, n) for name, (k, n) in counts.items()}
    estimates["F1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    intervals["F1"] = tuple(2 * q / (1 + q) for q in wilson(tp, tp + fp + fn))
    return estimates, intervals


def auc_intervals(y, probabilities, seed, *, overall=False):
    positive, negative = probabilities[y == 1], probabilities[y == 0]
    if not len(positive) or not len(negative):
        return np.full((2, probabilities.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    resamples = BOOTSTRAPS if overall else 5000
    chunks = []
    chunk_size = 500 if overall else resamples
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        pos = positive[rng.integers(len(positive), size=(count, len(positive)))]
        neg = negative[rng.integers(len(negative), size=(count, len(negative)))]
        differences = pos[:, :, None, :] - neg[:, None, :, :]
        chunks.append((differences > 0).mean(axis=(1, 2)) + .5 * (differences == 0).mean(axis=(1, 2)))
    return np.quantile(np.concatenate(chunks), [.025, .975], axis=0)


def brier_intervals(y, probabilities, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for label in (0, 1):
        positions = np.flatnonzero(y == label)
        if len(positions):
            draws.append(rng.choice(positions, size=(BOOTSTRAPS, len(positions)), replace=True))
    indices = np.concatenate(draws, axis=1)
    errors = (probabilities - y[:, None]) ** 2
    return np.quantile(errors[indices].mean(axis=1), [.025, .975], axis=0)


def citl(y, p):
    if set(y) != {0, 1} or not ((p > 0) & (p < 1)).all():
        return {"CITL": np.nan, "CITL_CI95_low": np.nan, "CITL_CI95_high": np.nan}
    offset = logit(p)
    score = lambda a: float(np.sum(y - expit(offset + a)))
    likelihood = lambda a: float(np.sum(y * (offset + a) - np.logaddexp(0, offset + a)))
    left, right = -1., 1.
    while score(left) < 0:
        left *= 2
    while score(right) > 0:
        right *= 2
    estimate = float(brentq(score, left, right))
    maximum = likelihood(estimate)
    equation = lambda a: 2 * (maximum - likelihood(a)) - chi2.ppf(.95, 1)
    width = 1.
    while equation(estimate - width) < 0:
        width *= 2
    low = float(brentq(equation, estimate - width, estimate))
    width = 1.
    while equation(estimate + width) < 0:
        width *= 2
    high = float(brentq(equation, estimate, estimate + width))
    return {"CITL": estimate, "CITL_CI95_low": low, "CITL_CI95_high": high}


def plot_calibration(out, y, p, n_bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
    assignments = np.searchsorted(edges[1:-1], p)
    rows = []
    for index in range(n_bins):
        mask = assignments == index
        if mask.any():
            low, high = wilson(int(y[mask].sum()), int(mask.sum()))
            rows.append({"bin": index + 1, "N": int(mask.sum()), "responder_N": int(y[mask].sum()),
                         "mean_probability": float(p[mask].mean()), "observed_response_fraction": float(y[mask].mean()),
                         "observed_fraction_CI95_low": low, "observed_fraction_CI95_high": high})
    bins = pd.DataFrame(rows)
    np.testing.assert_allclose(bins.mean_probability, predicted, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bins.observed_response_fraction, actual, rtol=0, atol=1e-12)
    assert bins.N.sum() == len(y)
    bins.to_csv(out / f"microbiome_calibration_quantile{n_bins}_bins.csv", index=False)
    plt.rcParams["svg.fonttype"] = "none"
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="#777D82", label="Ideal calibration")
    errors = np.vstack([actual - bins.observed_fraction_CI95_low, bins.observed_fraction_CI95_high - actual])
    axis.errorbar(predicted, actual, yerr=np.maximum(errors, 0), fmt="o-", capsize=3,
                  color="#258477", label="Microbiome-only (95% Wilson CI)")
    axis.set(xlim=(0, 1), ylim=(-.02, 1.03), xlabel="Mean predicted probability of response",
             ylabel="Observed response proportion", title=f"External calibration | N={len(y)}; {n_bins} quantile bins")
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(out / f"microbiome_calibration_quantile{n_bins}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def plot_discrimination(out, y, probabilities):
    if set(y) != {0, 1}:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ("#737B83", "#258477", "#D96E53")
    labels = ("Clinical-only", "Microbiome-only", "Combined")
    axes[0].plot([0, 1], [0, 1], "--", color="#A5A9AC", lw=1)
    axes[1].axhline(y.mean(), linestyle="--", color="#A5A9AC", lw=1, label=f"Response prevalence={y.mean():.3f}")
    for index, (label, color) in enumerate(zip(labels, colors)):
        p = probabilities[:, index]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=color, label=f"{label}: AUROC={roc_auc_score(y, p):.3f}")
        axes[1].plot(recall, precision, color=color, label=f"{label}: AP={average_precision_score(y, p):.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="External ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="External precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(-.02, 1.03)
        axis.legend(loc="lower right" if axis is axes[0] else "lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(out / f"external_roc_pr.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def evaluate_external(args, registry, out):
    predictions = pd.read_csv(args.predictions, dtype={"id": str}, float_precision="round_trip")
    meta = pd.read_csv(args.meta, sep="\t", dtype=str, usecols=["id", "group", "subtype", "biologic"])
    if meta.id.duplicated().any() or meta.isna().any().any() or not set(meta.group).issubset({"0", "1"}):
        raise ValueError("Invalid external outcome metadata")
    meta["biologic"] = meta.biologic.replace({"USK": "UST"})
    if not set(meta.subtype).issubset({"1", "2"}):
        raise ValueError("Subtype must be 1=UC, 2=CD")
    if set(predictions.model) != set(LABELS) or predictions.duplicated(["model", "id"]).any():
        raise ValueError("Expected exactly three models with unique patient predictions")
    aligned = {}
    for name in LABELS:
        subset = predictions[predictions.model.eq(name)].set_index("id")
        if set(subset.index) != set(meta.id):
            raise ValueError("Prediction and outcome IDs differ; no samples removed")
        subset = subset.loc[meta.id]
        if not np.isfinite(subset.probability).all() or not subset.probability.between(0, 1).all():
            raise ValueError("Invalid raw prediction probability")
        if "true_response" in subset:
            np.testing.assert_array_equal(subset.true_response.astype(int), meta.group.astype(int))
        for column in ("subtype", "biologic"):
            if column in subset:
                values = subset[column].astype(str).replace({"USK": "UST"}) if column == "biologic" else subset[column].astype(str)
                np.testing.assert_array_equal(values, meta[column])
        aligned[name] = subset.probability.to_numpy()
    y = meta.group.astype(int).to_numpy()
    probabilities = np.column_stack([aligned[name] for name in LABELS])
    scopes = [("All", np.ones(len(meta), dtype=bool)), ("UC", meta.subtype.eq("1").to_numpy()),
              ("CD", meta.subtype.eq("2").to_numpy())]
    scopes.extend((drug, meta.biologic.eq(drug).to_numpy()) for drug in sorted(meta.biologic.unique()))
    # Retain historical subgroup seed order regardless of alphabetical drug order.
    seed_index = {"All": 0, "UC": 1, "CD": 2, "AntiTNF": 3, "VDZ": 4, "UST": 5}
    rows, diagnostic_rows = [], []
    for scope, mask in scopes:
        if not mask.any():
            continue
        yy, pp = y[mask], probabilities[mask]
        index = seed_index.get(scope, 6)
        if scope == "All":
            ordering = np.argsort(meta.id.to_numpy(), kind="stable")
            auc_ci = auc_intervals(y[ordering], probabilities[ordering], 20261008, overall=True)
        else:
            auc_ci = auc_intervals(yy, pp, 20261009 + index)
        brier_ci = brier_intervals(yy, pp, 20261013 + index)
        both = set(yy) == {0, 1}
        for model_index, name in enumerate(LABELS):
            p = pp[:, model_index]
            threshold = registry[name]["threshold"]
            tn, fp, fn, tp = confusion_matrix(yy, p >= threshold, labels=[0, 1]).ravel()
            estimates, intervals = classification_intervals(int(tp), int(fp), int(tn), int(fn))
            row = {"model": name, "subgroup": scope, "N": len(yy), "responder_N": int(yy.sum()),
                   "nonresponder_N": int((yy == 0).sum()), "locked_public_cutoff": threshold,
                   "AUROC": float(roc_auc_score(yy, p)) if both else np.nan,
                   "AUROC_CI95_low": auc_ci[0, model_index], "AUROC_CI95_high": auc_ci[1, model_index],
                   "AUPRC_AP": float(average_precision_score(yy, p)) if both else np.nan,
                   "Brier_score": float(np.mean((p - yy) ** 2)),
                   "Brier_CI95_low": brier_ci[0, model_index], "Brier_CI95_high": brier_ci[1, model_index],
                   "mean_predicted_probability": float(p.mean()), "observed_response_rate": float(yy.mean()),
                   "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), **estimates}
            for metric, (low, high) in intervals.items():
                row.update({f"{metric}_CI95_low": low, f"{metric}_CI95_high": high})
            rows.append(row)
            if scope == "All":
                diagnostic_rows.append({"model": name, "slope_fixed": 1, **citl(yy, p)})
    performance = pd.DataFrame(rows)
    performance.to_csv(out / "external_subgroup_performance.csv", index=False)
    performance[performance.model.eq("microbiome_only")].to_csv(out / "microbiome_external_subgroup_performance.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(out / "external_citl.csv", index=False)
    for n_bins in (4, 5):
        plot_calibration(out, y, aligned["microbiome_only"], n_bins)
    plot_discrimination(out, y, probabilities)
    input_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in (("predictions", args.predictions), ("meta", args.meta), ("cutoffs", args.cutoffs))}
    audit = {"input_sha256": input_hashes, "prediction_model_fits": 0, "predictions_modified": False,
             "thresholds_optimized_or_modified": False, "diagnostic_CITL_intercepts_estimated": len(diagnostic_rows),
             "diagnostic_CITL_applied_to_predictions": False, "classification_CI": "Wilson; F1 Wilson indirect",
             "AUROC_CI": "paired response-stratified percentile bootstrap: 10000 overall, 5000 subgroup",
             "Brier_CI": "10000 paired response-stratified percentile bootstrap",
             "CITL_CI": "profile likelihood with fixed slope=1", "AUPRC": "Average Precision; no AP CI computed",
             "limitations": "Fixed-model patient-level uncertainty only; no selection uncertainty. Small subgroups and overlapping subtype/biologic strata are not independent validation."}
    (out / "evaluation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(performance[performance.subgroup.eq("All")].to_string(index=False))
