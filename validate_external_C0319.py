"""Evaluate the already locked C0319 model on the current Chinese split tables."""

import argparse
import csv
import io
import json
import logging
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

import logo_benchmark as b

ROOT = Path(__file__).resolve().parent
DEVELOPMENT = Path("outputs/development")
INPUT = Path("data/external")
MODEL_SHA256 = "20792321d7708688b46bc550840209f235c09ce8e6ef447210f1635fd43ac05d"
EXTERNAL_SHA256 = {
    "meta": "8f99f84646542996aea754616b6a116c7004a079a7421c0a74f8d84160fefa43",
    "species": "148db555ba30276beaf0fbe6bd2d058d1e5e21223bab5c4bcb823cdd82062bd6",
    "pwy": "b3ce127078757c3bddb6362f9d0a6a1e5e72a098013adf8defd4759a3c6ab3ea",
}


def load_tables(directory):
    frames, paths = {}, {}
    for block in EXTERNAL_SHA256:
        path = Path(directory) / f"yyyy_external_0928_pwy_matched_integrated_species_{block}.tsv"
        if b.core.sha(path.read_bytes()) != EXTERNAL_SHA256[block]:
            raise ValueError(f"Chinese {block} version changed since audit; stop, do not silently switch labels")
        raw = path.read_bytes()
        header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate headers: {block}")
        frames[block] = pd.read_csv(io.BytesIO(raw), sep="\t", dtype={"id": str})
        paths[block] = path
    return align_tables(frames), paths


def align_tables(frames):
    meta = frames["meta"].sort_values("id").reset_index(drop=True)
    if len(meta) != 52 or meta.id.isna().any() or meta.id.duplicated().any():
        raise ValueError("Exactly 52 unique identified external patients required; no silent deletion")
    matrices = {}
    for block in ("species", "pwy"):
        frame = frames[block]
        if frame.id.isna().any() or frame.id.duplicated().any() or set(frame.id) != set(meta.id):
            raise ValueError(f"External {block} IDs do not match metadata one-to-one")
        raw = frame.set_index("id").loc[meta.id].astype(float)
        if not all(n.startswith(block+"__") for n in raw):
            raise ValueError("Non-microbial column in external abundance matrix")
        if not np.isfinite(raw.to_numpy()).all() or (raw < 0).any().any():
            raise ValueError("Invalid abundance values; no imputation or removal")
        if (raw.sum(axis=1) <= 0).any():
            raise ValueError("Empty external abundance profile")
        if not np.allclose(raw.sum(axis=1), 1, atol=2e-5, rtol=0):
            raise ValueError("External input scale no longer matches audited relative-abundance profiles")
        matrices[block] = raw
    # Outcomes and clinical variables are deliberately not validated or used for inference.
    return meta, matrices["species"], matrices["pwy"]


def locked_model(development_dir=None):
    development = Path(development_dir or DEVELOPMENT)
    path = development / "final/Microbiome_LOGO_selected_final_175.joblib"
    if b.core.sha(path.read_bytes()) != MODEL_SHA256:
        raise ValueError("Locked public model changed")
    registry = json.loads((development/"final/model_registry.json").read_text())
    cutoff = json.loads((development/"locked_threshold.json").read_text())
    payload = joblib.load(path)
    if registry["model_SHA256"] != MODEL_SHA256 or payload["pipeline"]["candidate_id"] != "C0319":
        raise ValueError("Wrong locked model/version")
    if payload["threshold_nonresponse"] != cutoff["threshold_nonresponse"] or payload["classifier"].get_params() != registry["parameters"]:
        raise ValueError("Model/threshold/registry disagree")
    return payload, path


def feature_audit(payload, species, pwy):
    rows = []
    selected = set(payload["features"])
    for block, raw in (("Species", species), ("PWY", pwy)):
        for name in payload["preprocessing"][block]["retained_raw_features"]:
            present = name in raw
            if not present:
                raise ValueError(f"Missing required {block} CLR/scaling component: {name}; zero filling prohibited")
            a = raw[name]
            rows.append(dict(feature=name, block=block, selected_in_classifier=name in selected,
                present_in_external=True, imputed_zero=False, min=float(a.min()),median=float(a.median()),
                max=float(a.max()),number_nonzero=int((a>0).sum()),prevalence=float((a>0).mean())))
    return pd.DataFrame(rows)


def independent_design(payload, species, pwy):
    blocks = []
    for block, raw in (("Species",species),("PWY",pwy)):
        state = payload["preprocessing"][block]
        names = state["retained_raw_features"]
        a = raw[names].to_numpy(float)
        if block == "Species":
            a = np.log(a+state["pseudocount"])
            a -= a.mean(axis=1,keepdims=True)
        a = (a-np.asarray(state["center"]))/np.asarray(state["scale"])
        blocks.append(pd.DataFrame(a,index=raw.index,columns=names))
    return pd.concat(blocks,axis=1)[payload["features"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--development-dir", type=Path, default=DEVELOPMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or args.development_dir/"external_chinese_01"
    args.output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[
        logging.StreamHandler(),logging.FileHandler(args.output/"run.log")])
    payload, model_path = locked_model(args.development_dir)
    (meta,species,pwy),paths = load_tables(args.input_dir)
    snapshot = args.output/"input_snapshot"
    snapshot.mkdir()
    for path in paths.values():
        shutil.copy2(path,snapshot/path.name)
    shutil.copy2(Path(__file__),args.output/"validate_external_C0319_frozen.py")
    audit = feature_audit(payload,species,pwy)
    logging.info("C0319 locked. All52 align; all119 Species and388 PWY required components present; no imputation")
    # Only IDs enter inference. External labels and clinical variables are excluded.
    ids = meta[["id"]].copy()
    q = b.predict_final(payload,ids,species,pwy)
    x = b.core.design(ids,species,pwy,payload["preprocessing"],payload["features"])
    manual = independent_design(payload,species,pwy)
    np.testing.assert_allclose(x.to_numpy(),manual.to_numpy(),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(q,b.nonresponse_probability(payload["classifier"],manual))
    poisoned = ids.assign(group=999,subtype="INVALID",biologic="INVALID",age=-999,dataset="INVALID")
    np.testing.assert_array_equal(q,b.predict_final(payload,poisoned,species,pwy))
    threshold = payload["threshold_nonresponse"]
    pred = (q>=threshold).astype(int)
    predictions = pd.DataFrame(dict(id=meta.id,probability_nonresponse=q,probability_response=1-q,
        threshold_nonresponse=threshold,predicted_nonresponse=pred,
        predicted_label=np.where(pred==1,"Non-response","Response")))
    prediction_path = args.output/"external_predictions_before_outcome_join.csv"
    predictions.to_csv(prediction_path,index=False)
    b.core.dump(args.output/"prediction_freeze.json",dict(file=prediction_path.name,
        SHA256=b.core.sha(prediction_path.read_bytes()),model_SHA256=MODEL_SHA256,
        external_input_SHA256=EXTERNAL_SHA256,threshold_nonresponse=threshold,
        outcome_or_clinical_predictors_used=False,training=False,feature_selection=False,
        tuning=False,calibration=False,threshold_optimization=False))
    logging.info("Saved and hashed label-free predictions BEFORE outcome join; original raw predict_proba, no calibration")
    # Evaluation begins only after the label-free predictions have been frozen.
    if meta.group.isna().any() or set(meta.group)!={0,1}:
        raise ValueError("Invalid external outcome coding; frozen predictions retained but metrics not calculated")
    y = 1-meta.group.to_numpy(dtype=int)
    result = dict(model="Microbiome-only C0319",positive_event="Non-response (original group=0)",
        **b.discrimination(y,q),**b.classification(y,q,threshold),**b.distribution(q))
    pd.DataFrame([result]).to_csv(args.output/"external_performance.csv",index=False)
    with_labels = predictions.assign(original_group=meta.group,y_nonresponse=y,
        correct=pred==y)
    with_labels.to_csv(args.output/"external_predictions_with_outcomes.csv",index=False)
    audit.to_csv(args.output/"all_required_feature_coverage_and_raw_distribution.csv",index=False)
    selected_audit = audit.loc[audit.selected_in_classifier].set_index("feature").loc[payload["features"]].reset_index()
    selected_audit["transformed_min"] = x.min(axis=0).to_numpy()
    selected_audit["transformed_median"] = x.median(axis=0).to_numpy()
    selected_audit["transformed_max"] = x.max(axis=0).to_numpy()
    selected_audit["varies_across_external"] = x.nunique().gt(1).to_numpy()
    selected_audit["locked_model_MDI_importance"] = payload["classifier"].feature_importances_
    selected_audit.to_csv(args.output/"selected_16_feature_audit.csv",index=False)
    x.to_csv(args.output/"external_locked_transformed_52x16.tsv",sep="\t",index_label="id")
    pd.DataFrame([[result["TN"],result["FP"]],[result["FN"],result["TP"]]],
        index=["True_Response","True_Nonresponse"],columns=["Pred_Response","Pred_Nonresponse"]).to_csv(
        args.output/"confusion_matrix_nonresponse_positive.csv",index_label="true_label")
    for block,path in paths.items():
        if b.core.sha(path.read_bytes())!=EXTERNAL_SHA256[block]:
            raise ValueError("External input changed during evaluation")
    if b.core.sha(model_path.read_bytes())!=MODEL_SHA256:
        raise ValueError("Locked public model changed during evaluation")
    registry = dict(model_path=str(model_path),model_SHA256=MODEL_SHA256,
        external_input_paths={k:str(v) for k,v in paths.items()},external_SHA256=EXTERNAL_SHA256,
        selected_species=13,selected_PWY=3,required_CLR_Species=119,required_PWY=388,
        missing_required_features=0,imputed_columns=[],excluded_patients=[],N=52,
        label_mapping={"original_group_0":"Non-response","original_group_1":"Response"},
        clinical_variables_used=False,preprocessing_refitted=False,model_refitted=False,
        calibration=False,threshold_source="Locked public LOGO pooled OOF Youden; unchanged",
        threshold_nonresponse=threshold,manual_preprocessing_predictions_identical=True,
        poisoned_metadata_predictions_identical=True,model_and_source_hashes_unchanged=True,
        historical_external_reuse_warning="This cohort was examined previously in project history; this run does not make it prospectively untouched")
    b.core.dump(args.output/"external_validation_audit.json",registry)
    report = (
        "LOCKED C0319 EXTERNAL EVALUATION\n\n"
        "Final public175 ExtraTrees model:13 Species +3 PWY,700 trees,depth5,leaf2,sqrt,\n"
        "class_weight=None,seed20261125. No retraining,selection,tuning,calibration or threshold changes.\n"
        "SpeciesCLR over ALL119 public-retained Species, PWY original, public means/sampleSD applied.\n"
        "All119 Species and388 PWY required columns present; no zero imputation or patient exclusion.\n"
        "Input profiles already match the public0-1 relative-abundance scale; no /100 or renormalization.\n"
        "Only sample IDs and microbial tables entered inference. Age,subtype,biologic and group were not predictors.\n"
        "Label-free predictions were saved/hashed before outcome join; outcomes were used for evaluation only.\n"
        "Current metadata:52 patients,39 Response(original group1),13 Non-response(original group0).\n"
        +f"Non-response iff raw model probability >= {threshold:.17g}; cutoff locked from public OOF.\n"
        "AUROC/AP/Brier are threshold-independent; sensitivity/specificity/accuracy/PPV/NPV/F1 use that cutoff.\n"
        "AUPRC is average precision for Non-response; its prevalence baseline is13/52=.25.\n\n"
        +pd.DataFrame([result]).to_string(index=False)+"\n\n"
        "Both manual numpy preprocessing and poisoned-metadata inference reproduce predictions exactly.\n"
        "Model and all source hashes remained unchanged. No clinical or combined models were evaluated.\n"
        "Historical Chinese evaluations have already occurred; this is locked-model external-cohort evaluation,\n"
        "not a newly untouched prospective confirmation. Public selected LOGO performance remains an\n"
        "internally optimized development estimate, not unbiased nested-CV performance.\n\n"
        "Source metadata: "+str(paths["meta"])+"\n"
    )
    (args.output/"EXTERNAL_VALIDATION_REPORT.txt").write_text(report)
    logging.info("External N52: NR13 R39; AUROC=%.6f AP_NR=%.6f Brier=%.6f",result["AUROC"],result["nonresponse_AUPRC_AP"],result["Brier"])
    logging.info("Locked threshold %.17g: TP%d FP%d TN%d FN%d",threshold,result["TP"],result["FP"],result["TN"],result["FN"])


if __name__=="__main__":
    main()
