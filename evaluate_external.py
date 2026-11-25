"""Evaluate any already locked public LOGO pipeline, without fitting or tuning."""

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

import logo_benchmark as b
import validate_external_C0319 as source


def locked_payload(development):
    path=development/"final/Microbiome_LOGO_selected_final_175.joblib"
    registry=json.loads((development/"final/model_registry.json").read_text())
    threshold=json.loads((development/"locked_threshold.json").read_text())
    if b.core.sha(path.read_bytes())!=registry["model_SHA256"]:
        raise ValueError("Model SHA256 differs from its locked public registry")
    if threshold["source"]!="Pooled public LOGO OOF of selected pipeline" or threshold["Chinese_used"]:
        raise ValueError("Only a locked public-OOF threshold is allowed")
    payload=joblib.load(path)
    if payload["classifier"].get_params()!=registry["parameters"]:
        raise ValueError("Locked model parameters disagree with registry")
    if payload["threshold_nonresponse"]!=threshold["threshold_nonresponse"]:
        raise ValueError("Locked public cutoff changed")
    if payload["pipeline"]!=registry["pipeline"]:
        raise ValueError("Model/pipeline registry mismatch")
    return payload,path


def bootstrap_intervals(y,q,threshold,repetitions=5000,seed=20261125):
    if repetitions<100:
        raise ValueError("Use at least100 evaluation bootstrap replicates")
    rng=np.random.default_rng(seed)
    names=("AUROC","nonresponse_AUPRC_AP","Brier","sensitivity","specificity","accuracy","PPV","NPV","F1")
    saved={name:[] for name in names}
    y,q=np.asarray(y),np.asarray(q)
    for _ in range(repetitions):
        indices=rng.integers(0,len(y),len(y))
        yy,qq=y[indices],q[indices]
        if len(np.unique(yy))<2:
            continue
        metrics={**b.discrimination(yy,qq),**b.classification(yy,qq,threshold)}
        for name in names:
            if np.isfinite(metrics[name]):
                saved[name].append(metrics[name])
    rows=[]
    for name,values in saved.items():
        if not values:
            lo,hi=np.nan,np.nan
        else:
            lo,hi=np.quantile(values,[.025,.975])
        rows.append(dict(metric=name,CI95_lower=float(lo),CI95_upper=float(hi),valid_replicates=len(values),
            method="Patient-level unstratified percentile bootstrap; locked predictions/cutoff, no fitting"))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir",type=Path,default=Path("outputs/extratrees_v2"))
    parser.add_argument("--input-dir",type=Path,default=Path("data/external"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--bootstrap",type=int,default=5000)
    args=parser.parse_args()
    out=args.output or args.development_dir/"external"
    out.mkdir(parents=True,exist_ok=False)
    payload,path=locked_payload(args.development_dir)
    original_model_sha=b.core.sha(path.read_bytes())
    (meta,species,pwy),paths=source.load_tables(args.input_dir)
    audit=source.feature_audit(payload,species,pwy)
    snapshot=out/"input_snapshot"
    snapshot.mkdir()
    for p in paths.values():
        shutil.copy2(p,snapshot/p.name)
    shutil.copy2(Path(__file__),out/"evaluate_external_frozen.py")
    ids=meta[["id"]]
    q=b.predict_final(payload,ids,species,pwy)
    x=source.independent_design(payload,species,pwy)
    np.testing.assert_array_equal(q,b.nonresponse_probability(payload["classifier"],x))
    poisoned=ids.assign(group=999,subtype=999,biologic="IGNORED",age=-999,dataset="IGNORED")
    np.testing.assert_array_equal(q,b.predict_final(payload,poisoned,species,pwy))
    threshold=payload["threshold_nonresponse"]
    predictions=pd.DataFrame(dict(id=meta.id,probability_nonresponse=q,
        probability_response=1-q,predicted_nonresponse=(q>=threshold).astype(int),threshold_nonresponse=threshold))
    frozen=out/"external_predictions_before_outcome_join.csv"
    predictions.to_csv(frozen,index=False)
    b.core.dump(out/"prediction_freeze.json",dict(model_SHA256=original_model_sha,
        probability_file_SHA256=b.core.sha(frozen.read_bytes()),threshold_nonresponse=threshold,
        external_input_SHA256=source.EXTERNAL_SHA256,outcomes_used_for_inference=False,
        model_refitted=False,selection_or_tuning=False,calibration=False,threshold_modified=False))
    if meta.group.isna().any() or set(meta.group)!={0,1}:
        raise ValueError("Invalid response outcomes; only label-free probabilities were saved")
    y=1-meta.group.to_numpy(dtype=int)
    result=dict(model=payload["pipeline"]["candidate_id"],positive_event="Non-response",
        **b.discrimination(y,q),**b.classification(y,q,threshold),**b.distribution(q))
    pd.DataFrame([result]).to_csv(out/"external_performance.csv",index=False)
    bootstrap_intervals(y,q,threshold,args.bootstrap).to_csv(out/"external_performance_CI95.csv",index=False)
    predictions.assign(original_group=meta.group,y_nonresponse=y).to_csv(out/"external_predictions_with_outcomes.csv",index=False)
    audit.to_csv(out/"required_external_feature_audit.csv",index=False)
    x.to_csv(out/"external_transformed_matrix.tsv",sep="\t",index_label="id")
    b.core.dump(out/"probability_calibration_summary.json",dict(mean_probability_nonresponse=float(q.mean()),
        observed_nonresponse_rate=float(y.mean()),responder_mean_probability=float(q[y==0].mean()),
        nonresponder_mean_probability=float(q[y==1].mean()),calibration_method="None: original predict_proba",
        diagnostic_only=True,probabilities_modified=False))
    if b.core.sha(path.read_bytes())!=original_model_sha:
        raise ValueError("Locked model changed during external evaluation")
    for block,p in paths.items():
        if b.core.sha(p.read_bytes())!=source.EXTERNAL_SHA256[block]:
            raise ValueError("External source changed during evaluation")
    b.core.dump(out/"external_evaluation_audit.json",dict(model_hash_unchanged=True,external_sources_unchanged=True,
        required_features=len(audit),missing_required_features=0,imputed_columns=0,excluded_patients=0,
        prediction_before_outcome_join=True,metadata_predictors_used=False,
        caveat="Historically explored external cohort, not prospectively untouched confirmatory validation"))
    print(pd.DataFrame([result]).to_string(index=False))


if __name__=="__main__":
    main()
