"""Species + PWY model development with fixed three-group LOGO-CV."""

import argparse
import itertools
import json
import logging
from pathlib import Path
import platform
import shutil
import warnings

ROOT = Path(__file__).resolve().parent

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

import pipeline_core as core
import elastic_net_support as en

COUNTS = (4, 8, 12, 16, 20, 24, 32, 48, 64)
METHODS = ("association_F", "ElasticNet_nonzero_rank", "ExtraTrees_RFE")
SEED = 20261125
SELECTOR_EN = dict(C=.3, l1_ratio=.9)
EXPECTED_FOLDS = {"G1": (61, 114), "G2": (136, 39), "G3": (153, 22)}
SELECTION_RULE = (
    "Valid in all 3 folds; retain mean group AUROC within 0.01 of the maximum; "
    "then retain minimum group AUROC within 0.02 of the best in that pool; "
    "then fewer mean selected features, higher mean pairwise Jaccard, higher "
    "pooled nonresponse AP, higher mean AUROC, lower max tree depth, fewer trees, "
    "larger min_samples_leaf, candidate ID. "
    "No external data or classification threshold enters this selection rule."
)
LIMITATION = (
    "The SAME three LOGO folds are used for model selection and development evaluation. "
    "The selected CV performance is an internally optimized development estimate, "
    "NOT an unbiased nested-CV estimate. Held-out groups are excluded from fitting "
    "within each candidate fold, but their outcomes influence cross-candidate selection. "
    "The threshold is also optimized on these pooled development OOF predictions."
)


def configuration_grid():
    from run_extratrees_logo import configuration_grid as declared_grid
    return declared_grid()


def candidate_grid(configurations=None):
    supports = [(m, k) for m in METHODS for k in COUNTS] + [("All_retained", "ALL")]
    return [dict(candidate_id=f"C{i:04d}", method=m, feature_count=k, **cfg)
            for i, ((m, k), cfg) in enumerate(itertools.product(supports, configuration_grid() if configurations is None else configurations))]


def rank_features(names, values):
    names = np.asarray(names)
    order = np.lexsort((names, -np.nan_to_num(values, nan=0., posinf=1e300, neginf=-1e300)))
    return names[order].tolist()


def ranking_table(names, values, score_type):
    finite=np.nan_to_num(np.asarray(values),nan=0.,posinf=1e300,neginf=-1e300)
    lookup=dict(zip(names,finite))
    return [dict(rank=i,feature=n,score=float(lookup[n]),score_type=score_type)
            for i,n in enumerate(rank_features(names,finite),1)]


def make_supports(x, y, method, counts=COUNTS):
    """All inputs are the fold's training matrix/labels; no validation input accepted."""
    if not all(n.startswith(("species__", "pwy__")) for n in x) or x.columns.duplicated().any():
        raise ValueError("Selectors accept only uniquely named microbial predictors")
    result, diagnostic = {}, {}
    if method == "All_retained":
        return {"ALL": x.columns.tolist()}, diagnostic
    if method == "association_F":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores, _ = f_classif(x.to_numpy(), y)
        ranked = rank_features(x.columns, scores)
        diagnostic["training_ranking"]=ranking_table(x.columns,scores,"ANOVA_F")
    elif method == "ElasticNet_nonzero_rank":
        estimator, nonzero, status = en.fit_one(x, np.asarray(y), SELECTOR_EN)
        diagnostic = status
        if estimator is None:
            return {}, diagnostic
        keep = np.abs(estimator.coef_[0]) > en.COEFFICIENT_TOLERANCE
        ranked = rank_features(x.columns[keep], np.abs(estimator.coef_[0][keep]))
        diagnostic.update(nonzero_N=len(nonzero), C=SELECTOR_EN["C"], l1_ratio=SELECTOR_EN["l1_ratio"])
        diagnostic["training_ranking"]=ranking_table(x.columns[keep],np.abs(estimator.coef_[0][keep]),"absolute_nonzero_EN_coefficient")
    elif method == "ExtraTrees_RFE":
        current = x.columns.tolist()
        trace = []
        for k in sorted(counts, reverse=True):
            if k > x.shape[1]:
                continue
            while len(current) > k:
                model = ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2,
                    max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=1)
                model.fit(x[current].to_numpy(), y)
                ranked = rank_features(current, model.feature_importances_)
                next_n = max(k, min(len(current)-1, int(np.ceil(len(current)*.8))))
                wanted = set(ranked[:next_n])
                trace.append(dict(before=len(current), after=next_n,
                    training_ranking=ranking_table(current,model.feature_importances_,"selector_ExtraTrees_MDI")))
                current = [n for n in current if n in wanted]
            result[k] = current.copy()
        return result, dict(elimination_trace=trace, selector_trees=300, step_fraction=.2)
    else:
        raise ValueError(f"Unknown selector: {method}")
    for k in counts:
        if k <= len(ranked):
            wanted = set(ranked[:k])
            result[k] = [n for n in x if n in wanted]
    diagnostic["available_ranked_features"] = len(ranked)
    return result, diagnostic


def fit_classifier(candidate, x, y):
    if candidate["classifier"] != "ExtraTrees":
        raise ValueError("Expected an ExtraTrees classifier configuration")
    estimator = ExtraTreesClassifier(**candidate["parameters"])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x.to_numpy(), y)
    if any(issubclass(w.category, ConvergenceWarning) for w in seen):
        raise ValueError("Classifier did not converge; candidate invalid, no fallback")
    return estimator


def nonresponse_probability(model, x):
    index = np.flatnonzero(model.classes_ == 1)
    if len(index) != 1:
        raise ValueError("Missing Non-response class")
    return model.predict_proba(x.to_numpy())[:, int(index[0])]


def discrimination(y, q):
    y, q = np.asarray(y), np.asarray(q)
    return dict(N=len(y), nonresponse_N=int(y.sum()), response_N=int(len(y)-y.sum()),
        AUROC=float(roc_auc_score(y, q)), nonresponse_AUPRC_AP=float(average_precision_score(y, q)),
        Brier=float(np.mean((y-q)**2)), nonresponse_prevalence=float(y.mean()))


def distribution(q):
    q = np.asarray(q)
    return dict(probability_min=float(q.min()), probability_Q1=float(np.quantile(q,.25)),
        probability_median=float(np.median(q)), probability_Q3=float(np.quantile(q,.75)),
        probability_max=float(q.max()), probability_SD=float(q.std(ddof=1)), unique_probabilities=len(np.unique(q)))


def stability(sets):
    pairs = [len(a & b)/len(a | b) for a, b in itertools.combinations(sets, 2)]
    union = set.union(*sets)
    frequencies = {n: sum(n in s for s in sets) for n in union}
    return dict(mean_pairwise_Jaccard=float(np.mean(pairs)),
                **{f"features_{i}_of_3": sum(v == i for v in frequencies.values()) for i in (1,2,3)})


def choose(scores):
    pool = scores.loc[scores.status.eq("valid")].copy()
    if pool.empty:
        raise ValueError("No candidate valid in all three LOGO folds")
    pool = pool.loc[pool.mean_group_AUROC >= pool.mean_group_AUROC.max()-.01]
    pool = pool.loc[pool.min_group_AUROC >= pool.min_group_AUROC.max()-.02]
    return pool.sort_values(["mean_selected_features", "mean_pairwise_Jaccard", "pooled_nonresponse_AUPRC_AP",
        "mean_group_AUROC", "depth_complexity", "tree_complexity", "leaf_complexity", "candidate_id"],
        ascending=[True,False,False,False,True,True,False,True]).iloc[0]


def youden_threshold(y, q):
    fpr, tpr, thresholds = roc_curve(y, q, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    indices = np.flatnonzero(valid)
    # ROC thresholds descend: first maximum J chooses the highest threshold on ties.
    i = indices[np.argmax((tpr-fpr)[indices])]
    return float(thresholds[i]), pd.DataFrame(dict(threshold=thresholds, sensitivity=tpr,
        specificity=1-fpr, Youden_J=tpr-fpr, eligible_threshold=valid))


def classification(y, q, threshold):
    pred = np.asarray(q) >= threshold
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0,1]).ravel()]
    safe = lambda a,b: a/b if b else np.nan
    return dict(threshold_nonresponse=threshold, TP=tp, FP=fp, TN=tn, FN=fn,
        sensitivity=safe(tp,tp+fn), specificity=safe(tn,tn+fp), accuracy=safe(tp+tn,len(y)),
        PPV=safe(tp,tp+fp), NPV=safe(tn,tn+fn), F1=safe(2*tp,2*tp+fp+fn))


def predict_final(payload, meta, species, pwy):
    if payload["target"] != "Non-response = 1 - original group":
        raise ValueError("Unexpected model target")
    x = core.design(meta, species, pwy, payload["preprocessing"], payload["features"])
    return nonresponse_probability(payload["classifier"], x)


def main(configuration_provider=None, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output or Path("outputs/development"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/public"))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("audit", "data", "input_snapshot", "code_snapshot", "folds", "final"):
        (out/folder).mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(out/"run.log")])
    for source in (Path(__file__), Path(core.__file__), Path(en.__file__), ROOT/"test_logo_benchmark.py"):
        shutil.copy2(source, out/"code_snapshot"/source.name)
    if configuration_provider is not None:
        shutil.copy2(ROOT/"run_extratrees_logo.py",out/"code_snapshot/run_extratrees_logo.py")
    candidates = candidate_grid(configuration_provider() if configuration_provider else None)
    core.dump(out/"analysis_plan.json", dict(groups=core.GROUPS, expected_folds=EXPECTED_FOLDS,
        feature_counts=list(COUNTS)+["ALL"], selection_methods=list(METHODS)+["All_retained"],
        ElasticNet_selector=SELECTOR_EN, candidate_N=len(candidates), selection_rule=SELECTION_RULE,
        limitation=LIMITATION, positive_event="Non-response", Chinese_access_allowed=False,
        preprocessing="Training-only raw prevalence>=.20 and variance>1e-12; Species CLR over all retained Species; PWY original; training mean/sample SD ddof1",
        joint_feature_pool=True, forced_species_PWY_ratio=False,
        nested_CV=False, seed=SEED,
        candidate_manifest=candidates, versions=dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)))
    logging.info("Frozen plan: %d candidate pipelines, fixed 3-fold LOGO; NOT nested CV", len(candidates))
    meta, species, pwy = core.load_public(args.input_dir, out)
    y = 1-meta.group.to_numpy(dtype=int)
    groups = core.membership(meta).to_numpy()
    prepared, fold_audit, support_rows = [], [], []
    for group, tr, va in core.splits(meta):
        train, valid = meta.iloc[tr], meta.iloc[va]
        if (len(tr),len(va)) != EXPECTED_FOLDS[group]:
            raise ValueError(f"Unexpected fold size: {group}")
        if set(train.id)&set(valid.id) or set(train.dataset)&set(valid.dataset):
            raise ValueError("Sample or study leakage in fixed folds")
        target = out/"folds"/group
        target.mkdir()
        state = core.fit_preprocessing(train, species, pwy)
        xtr, xva = core.design(train,species,pwy,state), core.design(valid,species,pwy,state)
        supports = {}
        for method in list(METHODS)+["All_retained"]:
            available, diagnostic = make_supports(xtr, y[tr], method)
            supports[method] = available
            core.dump(target/f"{method}_diagnostic.json", diagnostic)
            for count, names in available.items():
                support_rows.extend(dict(validation_group=group, method=method, feature_count=count,
                    feature=n, block="Species" if n.startswith("species__") else "PWY") for n in names)
        core.dump(target/"preprocessing.json", state)
        core.dump(target/"training_provenance.json", dict(train_ids=train.id.tolist(), validation_ids=valid.id.tolist(),
            training_datasets=sorted(set(train.dataset)), validation_datasets=sorted(set(valid.dataset)),
            training_outcomes_nonresponse=y[tr].tolist(), input_sha256=core.EXPECTED_HASHES))
        prepared.append(dict(group=group,tr=tr,va=va,state=state,xtr=xtr,xva=xva,supports=supports))
        fold_audit.append(dict(validation_group=group, train_N=len(tr), validation_N=len(va),
            train_nonresponse_N=int(y[tr].sum()), validation_nonresponse_N=int(y[va].sum()),
            train_response_N=int(len(tr)-y[tr].sum()), validation_response_N=int(len(va)-y[va].sum()),
            training_datasets=";".join(sorted(set(train.dataset))), validation_datasets=";".join(sorted(set(valid.dataset))),
            sample_intersection=0, dataset_intersection=0,
            retained_species=len(state["Species"]["retained_raw_features"]), retained_PWY=len(state["PWY"]["retained_raw_features"])))
        logging.info("%s train=%d validate=%d: training-only preprocessing and selector supports ready", group,len(tr),len(va))
    pd.DataFrame(fold_audit).to_csv(out/"audit/fixed_LOGO_folds.csv",index=False)
    pd.DataFrame(support_rows).to_csv(out/"folds/all_training_selected_features.csv",index=False)
    score_rows, metrics_rows, predictions = [], [], []
    candidate_lookup = {c["candidate_id"]:c for c in candidates}
    for i,c in enumerate(candidates,1):
        row = {k:v for k,v in c.items() if k != "parameters"}
        row["parameters_json"] = json.dumps(c["parameters"],sort_keys=True)
        pars = c["parameters"]
        row.update(depth_complexity=pars.get("max_depth") or 100000,
                   tree_complexity=pars.get("n_estimators",0), leaf_complexity=pars.get("min_samples_leaf",0))
        missing = [f["group"] for f in prepared if c["feature_count"] not in f["supports"][c["method"]]]
        if missing:
            row.update(status="invalid",error="Requested count exceeds retained/nonzero selector features in: "+",".join(missing))
            score_rows.append(row)
            continue
        q = np.full(len(meta), np.nan)
        fold_rows, sets, sizes = [], [], []
        try:
            for fold in prepared:
                group,tr,va = fold["group"],fold["tr"],fold["va"]
                names = fold["supports"][c["method"]][c["feature_count"]]
                model = fit_classifier(c, fold["xtr"][names], y[tr])
                q[va] = nonresponse_probability(model,fold["xva"][names])
                sets.append(set(names)); sizes.append(len(names))
                species_N = sum(n.startswith("species__") for n in names)
                fr = dict(candidate_id=c["candidate_id"],validation_group=group,selected_species=species_N,
                    selected_PWY=len(names)-species_N,selected_features=len(names),
                    **discrimination(y[va],q[va]), **distribution(q[va]))
                fold_rows.append(fr)
            if not np.isfinite(q).all():
                raise ValueError("OOF predictions not complete")
        except ValueError as error:
            row.update(status="invalid",error=str(error))
            score_rows.append(row)
            logging.info("Invalid candidate %s: %s",c["candidate_id"],error)
            continue
        metrics_rows.extend(fold_rows)
        aucs = [f["AUROC"] for f in fold_rows]
        pooled = discrimination(y,q)
        row.update(status="valid",mean_group_AUROC=float(np.mean(aucs)), min_group_AUROC=float(min(aucs)),
            SD_group_AUROC=float(np.std(aucs,ddof=1)), pooled_AUROC=pooled["AUROC"],
            pooled_nonresponse_AUPRC_AP=pooled["nonresponse_AUPRC_AP"],pooled_Brier=pooled["Brier"],
            mean_group_nonresponse_AUPRC_AP=float(np.mean([f["nonresponse_AUPRC_AP"] for f in fold_rows])),
            mean_selected_features=float(np.mean(sizes)), **stability(sets))
        for fr in fold_rows:
            for key in ("AUROC","nonresponse_AUPRC_AP","selected_species","selected_PWY","selected_features"):
                row[fr["validation_group"]+"_"+key] = fr[key]
        score_rows.append(row)
        predictions.extend(dict(candidate_id=c["candidate_id"],id=meta.id.iloc[j],dataset=meta.dataset.iloc[j],
            validation_group=groups[j],original_group=int(meta.group.iloc[j]),y_nonresponse=int(y[j]),
            probability_nonresponse=float(q[j])) for j in range(len(meta)))
        if i%15 == 0:
            logging.info("Scored %d/%d candidate pipelines (each uses all 3 folds)",i,len(candidates))
            pd.DataFrame(score_rows).to_csv(out/"candidate_scores_progress.csv",index=False)
    scores = pd.DataFrame(score_rows)
    selected = choose(scores)
    winner = candidate_lookup[selected.candidate_id]
    scores["selected"] = scores.candidate_id.eq(selected.candidate_id)
    scores.to_csv(out/"candidate_pipeline_scores.csv",index=False)
    pd.DataFrame(metrics_rows).to_csv(out/"candidate_group_metrics.csv",index=False)
    oof = pd.DataFrame(predictions)
    oof.to_csv(out/"candidate_OOF_predictions.csv",index=False)
    chosen = oof.loc[oof.candidate_id.eq(selected.candidate_id)].copy()
    if len(chosen)!=175 or chosen.id.duplicated().any():
        raise ValueError("Chosen OOF coverage is not exactly one prediction per patient")
    q = chosen.probability_nonresponse.to_numpy()
    if chosen.id.tolist()!=meta.id.tolist():
        raise ValueError("Selected OOF row alignment changed")
    threshold,roc = youden_threshold(y,q)
    chosen["predicted_nonresponse"] = (q>=threshold).astype(int)
    chosen.to_csv(out/"selected_pipeline_OOF_predictions.csv",index=False)
    roc.to_csv(out/"selected_pipeline_OOF_ROC_Youden.csv",index=False)
    core.dump(out/"selected_pipeline.json", dict(candidate=winner, selection_rule=SELECTION_RULE,
        selected_scores={k:None if pd.isna(v) else v for k,v in selected.to_dict().items()}, interpretation=LIMITATION))
    core.dump(out/"locked_threshold.json", dict(threshold_nonresponse=threshold,decision="Non-response iff q >= threshold",
        source="Pooled public LOGO OOF of selected pipeline", tie_rule="Highest finite threshold among maximum-Youden-J thresholds",
        Chinese_used=False, default_0p5_used=False, internal_optimization_warning=LIMITATION))
    performance=[]
    for name,mask in [("Pooled",np.ones(len(y),dtype=bool))]+[(g,groups==g) for g in core.GROUPS]:
        performance.append(dict(evaluation=name,positive_event="Non-response", estimate_type="Internally optimized LOGO development estimate",
            **discrimination(y[mask],q[mask]),**classification(y[mask],q[mask],threshold),**distribution(q[mask])))
    pd.DataFrame(performance).to_csv(out/"selected_pipeline_development_performance.csv",index=False)
    feature_sets, selected_features = [], []
    for fold in prepared:
        names = fold["supports"][winner["method"]][winner["feature_count"]]
        feature_sets.append(set(names))
        selected_features.extend(dict(validation_group=fold["group"], feature=n,
            block="Species" if n.startswith("species__") else "PWY") for n in names)
        model = fit_classifier(winner,fold["xtr"][names],y[fold["tr"]])
        actual = nonresponse_probability(model,fold["xva"][names])
        np.testing.assert_array_equal(actual,q[fold["va"]])
        joblib.dump(dict(preprocessing=fold["state"],features=names,classifier=model,
            target="Non-response = 1 - original group", threshold=threshold,
            threshold_scope="Pooled public development OOF, NOT fold-training-only",pipeline=winner),
            out/"folds"/fold["group"]/"selected_development_fold_model.joblib")
    pd.DataFrame(selected_features).to_csv(out/"selected_pipeline_fold_features.csv",index=False)
    frequency_rows = [dict(feature=n,block="Species" if n.startswith("species__") else "PWY",
        selected_folds=";".join(g for g,s in zip(core.GROUPS,feature_sets) if n in s),
        fold_count=sum(n in s for s in feature_sets),frequency=sum(n in s for s in feature_sets)/3)
        for n in sorted(set.union(*feature_sets))]
    frequency = pd.DataFrame(frequency_rows).sort_values(["fold_count","feature"],ascending=[False,True])
    frequency.to_csv(out/"selected_pipeline_feature_stability.csv",index=False)
    frequency.groupby(["block","fold_count"]).size().rename("feature_N").reset_index().to_csv(
        out/"selected_pipeline_stability_summary.csv",index=False)
    logging.info("Selected %s: %s / %s / K=%s; fitting final model on all 175",winner["candidate_id"],
        winner["method"],winner["classifier"],winner["feature_count"])
    final_state = core.fit_preprocessing(meta,species,pwy)
    final_x = core.design(meta,species,pwy,final_state)
    final_supports,diagnostic = make_supports(final_x,y,winner["method"])
    if winner["feature_count"] not in final_supports:
        raise ValueError("Selected pipeline not feasible in full175 fit; no fallback or count modification")
    names = final_supports[winner["feature_count"]]
    final_model = fit_classifier(winner,final_x[names],y)
    payload = dict(preprocessing=final_state,features=names,classifier=final_model,
        target="Non-response = 1 - original group",threshold_nonresponse=threshold,
        pipeline=winner,input_hashes=core.EXPECTED_HASHES,limitation=LIMITATION)
    model_path=out/"final/Microbiome_LOGO_selected_final_175.joblib"
    joblib.dump(payload,model_path)
    np.testing.assert_array_equal(predict_final(payload,meta,species,pwy),
        predict_final(joblib.load(model_path),meta,species,pwy))
    core.dump(out/"final/preprocessing.json",final_state)
    core.dump(out/"final/selector_diagnostic.json",diagnostic)
    final_features=pd.DataFrame(dict(input_order=range(1,len(names)+1),feature=names,
        block=["Species" if n.startswith("species__") else "PWY" for n in names]))
    if hasattr(final_model,"feature_importances_"):
        final_features["MDI_importance"] = final_model.feature_importances_
    else:
        final_features["coefficient_nonresponse"] = final_model.coef_[0]
    final_features.to_csv(out/"final/final_selected_features.csv",index=False)
    final_x[names].to_csv(out/"final/final_training_matrix.tsv",sep="\t",index_label="id")
    core.dump(out/"final/model_registry.json",dict(model_SHA256=core.sha(model_path.read_bytes()),
        target=payload["target"],class_mapping={"0":"Response","1":"Non-response"},
        N=175,selected_species=sum(n.startswith("species__") for n in names),
        selected_PWY=sum(n.startswith("pwy__") for n in names), parameters=final_model.get_params(),
        threshold_nonresponse=threshold,threshold_source="Selected public LOGO OOF Youden",Chinese_accessed=False,
        source_hashes=core.EXPECTED_HASHES, pipeline=winner,limitation=LIMITATION))
    for block in core.EXPECTED_HASHES:
        path=args.input_dir/f"public_175_meta_species_pwy_raw_integrated_{block}.tsv"
        if core.sha(path.read_bytes())!=core.EXPECTED_HASHES[block]:
            raise ValueError("Source public inputs changed during execution")
    summary = (
        "FIXED THREE-GROUP LOGO DEVELOPMENT BENCHMARK\n\n"+LIMITATION+"\n\n"
        +f"Candidates declared: {len(candidates)}; valid in all 3 folds: {int(scores.status.eq('valid').sum())}.\n"
        +"Development predictors: integrated Species and PWY; validation units: predefined study groups.\n"
        +"All 175 retained, including stored whole-zero PWY profiles for SRR2145528/SRR8956534/SRR8956568;\n"
        +"these encodings are not confirmed biological pathway absence.\n"
        +"Within each candidate fold, raw prevalence/variance filters, CLR pseudocount/denominator, standardization,\n"
        +"feature ranking/RFE and classifier fitting use only the two training groups.\n"
        +"Species+PWY is one joint pool: no pre-fixed per-block counts or forcing both blocks to survive.\n"
        +"ElasticNet ranks only nonzero coefficients at C=.3, l1_ratio=.9; insufficient supports invalidate K.\n"
        +"ExtraTrees-RFE recursively drops up to20% per step to visit the predetermined count grid;\n"
        +"selector uses300 depth5 leaf2 sqrt trees; classifier parameters are in the frozen manifest.\n\n"
        +"Selection rule: "+SELECTION_RULE+"\n\nSelected pipeline:\n"+json.dumps(winner,indent=2)+"\n\n"
        +f"Final full175 signature: {sum(n.startswith('species__') for n in names)} Species + "
        +f"{sum(n.startswith('pwy__') for n in names)} PWY.\n"
        +f"Non-response threshold q >= {threshold:.17g}; locked from public OOF maximum Youden J.\n"
        +"Classification metrics below use the SAME pooled-development cutoff, including group rows.\n"
        +"AUPRC is average precision for NON-RESPONSE, not trapezoidal PR area; AUROC is threshold-independent.\n\n"
        +pd.DataFrame(performance).to_string(index=False)+"\n\n"
        +"Feature overlap is descriptive stability of development selections, not external validity.\n"
        +"Final-model importance is MDI (trees) or fitted logistic coefficient, not PLS VIP.\n"
        +"The all175 final fit has no independently evaluated performance here; no training AUC is reported.\n"
        +"Prior Chinese results have already been inspected historically; this run does not reset that history\n"
        +"or make the existing external cohort prospectively untouched.\n"
    )
    (out/"LOGO_DEVELOPMENT_REPORT.txt").write_text(summary)
    logging.info("Done: pooled AUROC=%.6f; nonresponse AP=%.6f; mean group AUROC=%.6f; min=%.6f",
        selected.pooled_AUROC, selected.pooled_nonresponse_AUPRC_AP,selected.mean_group_AUROC,selected.min_group_AUROC)
    logging.info("No Chinese access. All175 final model saved. %s",LIMITATION)


if __name__ == "__main__":
    main()
