#!/usr/bin/env python3
"""Portable reproduction of the frozen 47/48 grouped-CV prediction workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC

from diagnostics import classification_intervals, evaluate_external

HERE = Path(__file__).resolve().parent
GROUPS = {"G1": ["PRJNA685168"], "G2": ["SRP057027"],
          "G3": ["PRJNA384246", "PRJNA389280", "PRJNA398089"]}
MODELS = ("clinical_only", "microbiome_only", "combined")
REPRESENTATIONS = [("clr", "original"), ("clr", "rank"), ("log10", "log10"),
                   ("rank", "rank"), ("rank", "original"), ("binary", "binary")]
COUNTS = [(8, 4), (12, 6), (16, 6), (16, 8), (20, 6), (20, 10), (30, 6), (30, 10)]
SCREEN_MODELS = [
    *[{"classifier": "extra_trees", "n_estimators": 1200, "max_depth": depth,
       "min_samples_leaf": 2, "max_features": "sqrt"} for depth in (3, 4, 5)],
    {"classifier": "rf", "n_estimators": 1000, "max_depth": 4, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"classifier": "gb", "n_estimators": 120, "learning_rate": .03, "max_depth": 1, "subsample": .8},
    {"classifier": "logistic_l2", "C": .1}, {"classifier": "svm_linear", "C": .1},
]
CLINICAL = {"C": 1., "solver": "liblinear", "class_weight": "balanced", "max_iter": 5000}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {path}")
    path.mkdir(parents=True)
    return path


def read_table(path, *, separator="\t", columns=None):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle, delimiter=separator))
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names: {path}")
    if columns and not set(columns).issubset(header):
        raise ValueError(f"Missing required columns: {sorted(set(columns) - set(header))}")
    return pd.read_csv(path, sep=separator, dtype=str, usecols=columns)


def read_cohort(meta_path, species_path, pwy_path, *, public=False, clinical=True, outcome=True, expected_n=None):
    required = ["id"] + (["dataset"] if public else []) + (["group"] if outcome else [])
    if clinical:
        required += ["subtype", "biologic"]
    meta = read_table(meta_path, columns=required)
    if meta.isna().any().any() or meta.eq("").any().any() or meta.id.duplicated().any():
        raise ValueError("Metadata contains missing values or duplicate sample IDs; no rows removed")
    if expected_n is not None and len(meta) != expected_n:
        raise ValueError(f"Expected {expected_n} samples, found {len(meta)}")
    if outcome:
        if not set(meta.group).issubset({"0", "1"}):
            raise ValueError("Outcome coding must be group=1 responder, group=0 nonresponder")
        meta["response_binary"] = meta.group.astype(int)
    if public:
        if set(meta.dataset) != {dataset for datasets in GROUPS.values() for dataset in datasets}:
            raise ValueError("Public metadata must contain exactly the five predefined studies")
        if set(meta.response_binary) != {0, 1}:
            raise ValueError("Public development requires both outcome classes")
    if clinical:
        meta["biologic"] = meta.biologic.replace({"USK": "UST"})
        if not set(meta.subtype).issubset({"1", "2"}):
            raise ValueError("Subtype coding must be 1=UC, 2=CD")
    matrices = []
    for path in (species_path, pwy_path):
        frame = read_table(path)
        if "id" not in frame or frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f"Invalid matrix sample IDs: {path}")
        frame = frame.set_index("id").astype(float)
        if set(frame.index) != set(meta.id):
            raise ValueError("Metadata/Species/PWY IDs must match one-to-one; no silent intersection")
        if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
            raise ValueError("Abundances must be finite and nonnegative")
        matrices.append(frame.loc[meta.id])
    return meta, *matrices


def transform(raw, representation, pseudocount):
    if representation == "original":
        return raw.copy()
    if representation == "clr":
        logged = np.log(raw + pseudocount)
        return logged.sub(logged.mean(axis=1), axis=0)
    if representation == "log10":
        return np.log10(raw + pseudocount)
    if representation == "rank":
        return raw.rank(axis=1, method="average", pct=True)
    if representation == "binary":
        return (raw > 0).astype(float)
    raise ValueError(representation)


def block_state(raw, representation, prefix):
    retained = list(raw.columns[((raw > 0).mean(axis=0) >= .20) & (raw.var(axis=0) > 1e-12)])
    if not retained:
        raise ValueError("No features remain after training-only filtering")
    aligned = raw[retained]
    positive = aligned.to_numpy()
    positive = positive[(positive > 0) & np.isfinite(positive)]
    pseudocount = float(positive.min() / 2) if positive.size else 1e-8
    values = transform(aligned, representation, pseudocount)
    scale = values.std(axis=0, ddof=1).replace(0, 1).replace([np.inf, -np.inf], 1).fillna(1)
    return {"representation": representation, "prefix": prefix, "retained_raw_features": retained,
            "pseudocount": pseudocount, "center": values.mean(axis=0).tolist(), "scale": scale.tolist(),
            "prevalence_cutoff": .20, "variance_epsilon": 1e-12}


def apply_block(raw, state, allow_missing=False):
    columns = state["retained_raw_features"]
    missing = set(columns) - set(raw.columns)
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} training-retained columns missing. Confirm compatible complete profiles before explicitly allowing raw-zero filling.")
    aligned = raw.reindex(columns=columns, fill_value=0)
    values = transform(aligned, state["representation"], state["pseudocount"])
    values = ((values - pd.Series(state["center"], index=columns)) /
              pd.Series(state["scale"], index=columns)).replace([np.inf, -np.inf], 0).fillna(0)
    values.columns = [f"{state['prefix']}__{name}" for name in columns]
    return values


def association_rank(x, y, datasets):
    y, datasets = np.asarray(y), np.asarray(datasets)
    rows = []
    for feature in x.columns:
        effects, weights = [], []
        for dataset in sorted(set(datasets)):
            mask = datasets == dataset
            labels = y[mask]
            if len(np.unique(labels)) < 2:
                continue
            values = x.loc[mask, feature].to_numpy()
            effects.append(float(values[labels == 1].mean() - values[labels == 0].mean()))
            weights.append(int(mask.sum()))
        if effects:
            effects = np.asarray(effects)
            effect = float(np.average(effects, weights=weights))
            sign = np.sign(effect) if effect else 0
            consistency = float(np.mean(np.sign(effects) == sign)) if sign else 0.
            heterogeneity = float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.
        else:
            effect, consistency, heterogeneity = 0., 0., 0.
        rows.append({"feature": feature, "effect": effect, "direction_consistency": consistency,
                     "heterogeneity": heterogeneity, "assoc_score": abs(effect) * consistency / (1 + heterogeneity)})
    return pd.DataFrame(rows).sort_values(["assoc_score", "feature"], ascending=[False, True]).reset_index(drop=True)


def learn_design(meta, species, pwy, spec):
    state = {"clinical_levels": {column: sorted(meta[column].unique().tolist())
                                 for column in ("subtype", "biologic") if column in meta}}
    selections = []
    for block, raw, prefix, representation, count in (
            ("species", species, "species", spec["species_rep"], spec["n_species"]),
            ("pwy", pwy, "pwy", spec["pwy_rep"], spec["n_pwy"])):
        state[block] = block_state(raw.loc[meta.id], representation, prefix)
        x = apply_block(raw.loc[meta.id], state[block])
        ranked = association_rank(x, meta.response_binary.to_numpy(), meta.dataset.to_numpy())
        if len(ranked) < count:
            raise ValueError("Insufficient training-retained features for requested subset size")
        selections.append(ranked.head(count).assign(feature_type="Species" if block == "species" else "PWY"))
    return state, pd.concat(selections, ignore_index=True)


def clinical_features(state):
    return [f"clinical__{column}_{level}" for column, levels in state["clinical_levels"].items() for level in levels[1:]]


def design(meta, species, pwy, state, features, allow_missing=False):
    ids = meta.id.tolist()
    clinical = {}
    for column, levels in state.get("clinical_levels", {}).items():
        if any(feature.startswith(f"clinical__{column}_") for feature in features):
            for level in levels[1:]:
                clinical[f"clinical__{column}_{level}"] = (meta[column].to_numpy() == level).astype(float)
    blocks = [pd.DataFrame(clinical, index=ids)]
    for block in ("species", "pwy"):
        if any(feature.startswith(block + "__") for feature in features):
            blocks.append(apply_block({"species": species, "pwy": pwy}[block].loc[ids], state[block], allow_missing))
    matrix = pd.concat(blocks, axis=1)[features]
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Predictor matrix contains nonfinite values")
    return matrix


def model_factory(params, seed):
    arguments = {key: value for key, value in params.items() if key != "classifier"}
    name = params["classifier"]
    if name in ("extra_trees", "rf"):
        factory = ExtraTreesClassifier if name == "extra_trees" else RandomForestClassifier
        return factory(**arguments, class_weight="balanced" if name == "extra_trees" else "balanced_subsample",
                       random_state=seed, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(**arguments, random_state=seed)
    if name == "logistic_l2":
        return LogisticRegression(**arguments, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)
    if name == "svm_linear":
        # Historical screen uses sklearn's internal probability estimation; not strict nested tuning.
        return SVC(**arguments, kernel="linear", probability=True, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def point_metrics(y, p, cutoff=None):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    both = set(y) == {0, 1}
    result = {"N": len(y), "AUROC": float(roc_auc_score(y, p)) if both else np.nan,
              "AUPRC_AP": float(average_precision_score(y, p)) if both else np.nan,
              "Brier_score": float(np.mean((p - y) ** 2))}
    if cutoff is not None:
        tn, fp, fn, tp = confusion_matrix(y, p >= cutoff, labels=[0, 1]).ravel()
        result.update(dict(zip(("TP", "FP", "TN", "FN"), map(int, (tp, fp, tn, fn)))))
        estimates, _ = classification_intervals(int(tp), int(fp), int(tn), int(fn))
        result.update(estimates)
    return result


def youden_threshold(y, p):
    if set(y) != {0, 1}:
        raise ValueError("Youden selection requires both public outcome classes")
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.flatnonzero(np.isfinite(thresholds))
    winner = finite[np.argmax((tpr - fpr)[finite])]
    return float(thresholds[winner])


def folds(meta):
    for index, (name, test_datasets) in enumerate(GROUPS.items()):
        train = meta.loc[~meta.dataset.isin(test_datasets)].copy()
        test = meta.loc[meta.dataset.isin(test_datasets)].copy()
        if set(train.id) & set(test.id) or set(train.dataset) & set(test.dataset):
            raise ValueError("Grouped train/test leakage")
        if set(train.response_binary) != {0, 1} or set(test.response_binary) != {0, 1}:
            raise ValueError("A predefined fold does not have both response classes; do not modify folds")
        yield index, name, train, test


def evaluate_spec(meta, species, pwy, spec, seed):
    predictions, metrics = [], []
    for index, name, train, test in folds(meta):
        state, selected = learn_design(train, species, pwy, spec)
        features = selected.feature.tolist()
        model = model_factory(spec["params"], seed + index)
        model.fit(design(train, species, pwy, state, features), train.response_binary)
        p = model.predict_proba(design(test, species, pwy, state, features))[:, 1]
        frame = test[["id", "dataset"]].assign(outer_fold=name, true_response=test.response_binary, probability=p)
        predictions.append(frame)
        metrics.append(point_metrics(test.response_binary, p))
    frame = pd.concat(predictions, ignore_index=True)
    return {"mean_fold_AUROC": float(np.mean([row["AUROC"] for row in metrics])),
            "min_fold_AUROC": float(min(row["AUROC"] for row in metrics)),
            "pooled_AUROC": float(roc_auc_score(frame.true_response, frame.probability)),
            "pooled_AUPRC": float(average_precision_score(frame.true_response, frame.probability)),
            **{f"{name}_AUROC": row["AUROC"] for name, row in zip(GROUPS, metrics)}}


def select_candidates(scores):
    near = scores[scores.mean_fold_AUROC >= scores.mean_fold_AUROC.max() - .01]
    robust = near[near.min_fold_AUROC >= near.min_fold_AUROC.max() - .03]
    pool = robust[robust.mean_fold_AUROC >= robust.mean_fold_AUROC.max() - .01].copy()
    pool["n_total"] = pool.n_species + pool.n_pwy
    return pool.sort_values(["n_total", "mean_fold_AUROC", "pooled_AUROC"], ascending=[True, False, False]).iloc[0]


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("outer_groups") != GROUPS:
        raise ValueError("Configuration must preserve the predefined G1/G2/G3 groups")
    return config


def public_data(args, clinical):
    return read_cohort(args.meta, args.species, args.pwy, public=True, clinical=clinical,
                       outcome=True, expected_n=args.expected_n)


def run_search(args):
    meta, species, pwy = public_data(args, clinical=False)
    if args.command == "screen":
        grid = [{"species_rep": sp, "pwy_rep": pw, "n_species": ns, "n_pwy": npwy, "params": params}
                for sp, pw in REPRESENTATIONS for ns, npwy in COUNTS for params in SCREEN_MODELS]
        config = read_config(args.config)
    else:
        config = read_config(args.screen_config)
        original = config["spec"]
        if original["params"]["classifier"] != "extra_trees":
            raise ValueError("This published stage-2 grid refines ExtraTrees only; do not silently force another screen winner")
        grid = [{**original, "params": {"classifier": "extra_trees", "n_estimators": trees, "max_depth": depth,
                                       "min_samples_leaf": leaf, "max_features": fraction}}
                for trees in (700, 1200) for depth in (2, 3, 4, 5, 6, None)
                for leaf in (2, 3, 5) for fraction in ("sqrt", .5, 1.)]
    fingerprints = {key: sha256(getattr(args, key)) for key in ("meta", "species", "pwy")}
    if args.command == "tune" and config.get("public_input_sha256") != fingerprints:
        raise ValueError("Public inputs differ from stage 1")
    out = new_output(args.output)
    write_json(out / "protocol.json", {"stage": args.command, "candidate_count": len(grid), "outer_groups": GROUPS,
               "public_input_sha256": fingerprints, "external_inputs_allowed": False,
               "validation_status": "grouped-CV development; not unbiased strict nested-CV validation"})
    scores = []
    for index, spec in enumerate(grid, 1):
        print(f"{args.command}: {index}/{len(grid)}", flush=True)
        scores.append({"candidate_id": index, **{key: spec[key] for key in ("species_rep", "pwy_rep", "n_species", "n_pwy")},
                       "params_json": json.dumps(spec["params"], sort_keys=True),
                       **evaluate_spec(meta, species, pwy, spec, config["public_fold_seed_start"])})
    table = pd.DataFrame(scores)
    table.to_csv(out / "candidate_scores.csv", index=False)
    selected = select_candidates(table)
    config["spec"] = {key: selected[key] for key in ("species_rep", "pwy_rep", "n_species", "n_pwy")}
    config["spec"].update(n_species=int(selected.n_species), n_pwy=int(selected.n_pwy), params=json.loads(selected.params_json))
    config["public_input_sha256"] = fingerprints
    write_json(out / "selected_config.json", config)
    print(selected.to_string())


def run_oof(args):
    config = read_config(args.config)
    meta, species, pwy = public_data(args, clinical=True)
    out = new_output(args.output)
    frames, selected_rows, level_rows = [], [], []
    for index, fold, train, test in folds(meta):
        state, selected = learn_design(train, species, pwy, config["spec"])
        selected_rows.append(selected.assign(outer_fold=fold))
        for column, levels in state["clinical_levels"].items():
            for level in sorted(set(test[column]) - set(levels)):
                level_rows.append({"fold": fold, "variable": column, "test_only_level": level,
                                   "encoding": "all-zero reference encoding; historical behavior retained"})
        for name in MODELS:
            features = ([] if name == "clinical_only" else selected.feature.tolist())
            if name != "microbiome_only":
                features = clinical_features(state) + features
            model = (LogisticRegression(**CLINICAL, random_state=config["public_fold_seed_start"] + index)
                     if name == "clinical_only" else model_factory(config["spec"]["params"], config["public_fold_seed_start"] + index))
            model.fit(design(train, species, pwy, state, features), train.response_binary)
            p = model.predict_proba(design(test, species, pwy, state, features))[:, 1]
            frames.append(test[["id", "dataset"]].assign(model=name, outer_fold=fold,
                          true_response=test.response_binary, probability=p))
    predictions = pd.concat(frames, ignore_index=True)
    registry, summaries, fold_metrics = {}, [], []
    for name in MODELS:
        values = predictions[predictions.model.eq(name)]
        cutoff = youden_threshold(values.true_response.to_numpy(), values.probability.to_numpy())
        registry[name] = {"source": "public_oof", "threshold": cutoff}
        summaries.append({"model": name, "cutoff": cutoff, **point_metrics(values.true_response, values.probability, cutoff)})
        for fold, group in values.groupby("outer_fold", sort=False):
            fold_metrics.append({"model": name, "outer_fold": fold, **point_metrics(group.true_response, group.probability, cutoff)})
    write_json(out / "public_oof_thresholds.json", registry)
    write_json(out / "config.json", config)
    write_json(out / "input_hashes.json", {key: sha256(getattr(args, key)) for key in ("meta", "species", "pwy")})
    predictions.to_csv(out / "public_oof_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "public_pooled_performance.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "public_fold_performance.csv", index=False)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(out / "selected_features_by_fold.csv", index=False)
    stability = selected.groupby(["feature", "feature_type"], as_index=False).agg(n_selected_folds=("outer_fold", "nunique"))
    stability["selection_frequency"] = stability.n_selected_folds / 3
    stability.to_csv(out / "feature_stability.csv", index=False)
    pd.DataFrame(level_rows, columns=["fold", "variable", "test_only_level", "encoding"]).to_csv(out / "unseen_clinical_levels.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


def load_cutoffs(path):
    registry = json.loads(Path(path).read_text())
    for name in MODELS:
        entry = registry[name]
        if entry["source"] != "public_oof" or not np.isfinite(entry["threshold"]) or not 0 <= entry["threshold"] <= 1:
            raise ValueError("All thresholds must be locked public OOF thresholds")
    return registry


def run_fit(args):
    config = read_config(args.config)
    registry = load_cutoffs(args.cutoffs)
    meta, species, pwy = public_data(args, clinical=True)
    provenance = Path(args.cutoffs).parent / "input_hashes.json"
    if not provenance.is_file() or not (Path(args.cutoffs).parent / "config.json").is_file():
        raise ValueError("Final fitting requires OOF-generated config.json and input_hashes.json alongside cutoffs")
    if read_config(Path(args.cutoffs).parent / "config.json") != config:
        raise ValueError("Final configuration differs from OOF threshold configuration")
    fingerprints = {key: sha256(getattr(args, key)) for key in ("meta", "species", "pwy")}
    if json.loads(provenance.read_text()) != fingerprints:
        raise ValueError("Public inputs changed after OOF threshold development")
    out = new_output(args.output)
    state, selected = learn_design(meta, species, pwy, config["spec"])
    for name in MODELS:
        features = [] if name == "clinical_only" else selected.feature.tolist()
        if name != "microbiome_only":
            features = clinical_features(state) + features
        if name == "clinical_only":
            model = LogisticRegression(**CLINICAL, random_state=config["clinical_final_seed"])
            spec = CLINICAL
        else:
            model = model_factory(config["spec"]["params"], config["microbiome_final_seed"])
            spec = config["spec"]
        model.fit(design(meta, species, pwy, state, features), meta.response_binary)
        payload = {"model": model, "spec": spec, "features": features, "preprocessing": state,
                   "cutoff": registry[name]["threshold"], "sklearn_version": sklearn.__version__}
        path = out / f"{name}_complete_pipeline.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        registry[name]["final_model_sha256"] = sha256(path)
    selected.to_csv(out / "final_selected_microbial_features.csv", index=False)
    write_json(out / "public_oof_thresholds.json", registry)
    write_json(out / "config.json", config)
    write_json(out / "public_input_hashes.json", fingerprints)
    print("Fitted three final public models; no external data were read.")


def load_trusted_payload(path, entry, trusted):
    if not trusted:
        raise ValueError("Pickle can execute code. Use --trust-model-files only for your own trusted model files.")
    if sha256(path) != entry.get("final_model_sha256"):
        raise ValueError("Model hash differs from cutoff registry")
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if "sklearn_version" in payload and payload["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the sklearn version recorded in the fitted model")
    return payload


def run_predict(args):
    registry = load_cutoffs(args.cutoffs)
    # Outcomes and dataset labels are deliberately not read in prediction mode.
    meta, species, pwy = read_cohort(args.meta, args.species, args.pwy, clinical=True, outcome=False)
    out = new_output(args.output)
    frames, availability, levels = [], [], []
    for name in MODELS:
        path = Path(args.models) / f"{name}_complete_pipeline.pkl"
        payload = load_trusted_payload(path, registry[name], args.trust_model_files)
        state, features = payload["preprocessing"], payload["features"]
        matrix = design(meta, species, pwy, state, features, args.allow_missing_microbial_columns)
        if list(payload["model"].feature_names_in_) != list(matrix.columns):
            raise ValueError("Fitted and reconstructed feature order differ")
        if list(payload["model"].classes_) != [0, 1]:
            raise ValueError("Fitted class order must be 0=nonresponder, 1=responder")
        p = payload["model"].predict_proba(matrix)[:, 1]
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Fitted model returned invalid probabilities")
        cutoff = registry[name]["threshold"]
        frames.append(meta[["id", "subtype", "biologic"]].assign(model=name, probability=p,
                      locked_public_cutoff=cutoff, predicted_response=(p >= cutoff).astype(int)))
        for column, training_levels in state.get("clinical_levels", {}).items():
            if name != "microbiome_only":
                for level in sorted(set(meta[column]) - set(training_levels)):
                    levels.append({"model": name, "variable": column, "unseen_external_level": level})
        for block, raw in (("species", species), ("pwy", pwy)):
            if not any(feature.startswith(block + "__") for feature in features):
                continue
            for raw_feature in state[block]["retained_raw_features"]:
                selected = block + "__" + raw_feature in features
                availability.append({"model": name, "block": block, "raw_feature": raw_feature,
                                     "selected": selected, "present_in_external": raw_feature in raw.columns,
                                     "raw_zero_filled": raw_feature not in raw.columns})
    pd.concat(frames, ignore_index=True).to_csv(out / "external_predictions.csv", index=False)
    pd.DataFrame(availability).to_csv(out / "external_feature_availability.csv", index=False)
    pd.DataFrame(levels, columns=["model", "variable", "unseen_external_level"]).to_csv(out / "unseen_clinical_levels.csv", index=False)
    write_json(out / "prediction_audit.json", {"fits_performed": 0, "outcome_columns_read": False,
               "raw_predict_proba": True, "calibration_applied": False, "thresholds_modified": False,
               "model_hashes": {name: registry[name]["final_model_sha256"] for name in MODELS},
               "input_sha256": {key: sha256(getattr(args, key)) for key in ("meta", "species", "pwy")},
               "missing_raw_columns_explicitly_allowed": args.allow_missing_microbial_columns})
    print("Saved frozen raw predict_proba outputs; external outcomes were not read.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "tune", "public-oof", "fit-final", "predict", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run", action="store_true", help="Explicitly execute; otherwise check arguments only")
        command.add_argument("--output", type=Path, required=True)
        if name != "evaluate":
            for field in ("meta", "species", "pwy"):
                command.add_argument("--" + field, type=Path, required=True)
        if name in ("screen", "public-oof", "fit-final"):
            command.add_argument("--config", type=Path, default=HERE / "configs/locked_v48.json")
        if name in ("screen", "tune", "public-oof", "fit-final"):
            command.add_argument("--expected-n", type=int, default=175)
        if name == "tune":
            command.add_argument("--screen-config", type=Path, required=True)
        if name in ("fit-final", "predict", "evaluate"):
            command.add_argument("--cutoffs", type=Path, required=True)
        if name == "predict":
            command.add_argument("--models", type=Path, required=True)
            command.add_argument("--trust-model-files", action="store_true")
            command.add_argument("--allow-missing-microbial-columns", action="store_true")
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        print("Check only. No cohort read, model fitting, prediction or file writing. Add --run to execute.")
        return
    if args.command in ("screen", "tune"):
        run_search(args)
    elif args.command == "public-oof":
        run_oof(args)
    elif args.command == "fit-final":
        run_fit(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        evaluate_external(args, load_cutoffs(args.cutoffs), new_output(args.output))


if __name__ == "__main__":
    main()
