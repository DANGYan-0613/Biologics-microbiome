"""Recalculate saved external metrics and reproduce locked-model predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import evaluate_external as evaluation
import external_inputs as inputs
import logo_benchmark as b


def verify(development,input_dir,output):
    payload,model_path=evaluation.locked_payload(development)
    (meta,species,pwy),paths=inputs.load_tables(input_dir)
    freeze=json.loads((output/"prediction_freeze.json").read_text())
    file=output/"external_predictions_before_outcome_join.csv"
    if b.core.sha(file.read_bytes())!=freeze["probability_file_SHA256"]:
        raise ValueError("Saved prediction hash mismatch")
    if b.core.sha(model_path.read_bytes())!=freeze["model_SHA256"]:
        raise ValueError("Frozen model hash mismatch")
    predictions=pd.read_csv(file,float_precision="round_trip")
    scored=pd.read_csv(output/"external_predictions_with_outcomes.csv",float_precision="round_trip")
    performance=pd.read_csv(output/"external_performance.csv",float_precision="round_trip").iloc[0]
    if predictions.id.tolist()!=meta.id.tolist() or len(predictions)!=52 or predictions.id.duplicated().any():
        raise ValueError("External prediction/sample alignment mismatch")
    if {"group","original_group","y_nonresponse","age","subtype","biologic"}&set(predictions.columns):
        raise ValueError("Label-free prediction file contains outcome/clinical columns")
    threshold=payload["threshold_nonresponse"]
    assert threshold==freeze["threshold_nonresponse"]
    np.testing.assert_array_equal(predictions.threshold_nonresponse,np.full(len(meta),threshold))
    q=predictions.probability_nonresponse.to_numpy()
    np.testing.assert_array_equal(q,b.predict_final(payload,meta[["id"]],species,pwy))
    np.testing.assert_array_equal(q,b.nonresponse_probability(payload["classifier"],inputs.independent_design(payload,species,pwy)))
    y=1-meta.group.to_numpy(dtype=int)
    np.testing.assert_array_equal(scored.id,predictions.id)
    np.testing.assert_array_equal(scored.y_nonresponse,y)
    np.testing.assert_array_equal(scored.probability_nonresponse,q)
    pred=q>=threshold
    np.testing.assert_array_equal(pred,predictions.predicted_nonresponse)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    safe=lambda a,z:a/z if z else np.nan
    expected=dict(AUROC=roc_auc_score(y,q),nonresponse_AUPRC_AP=average_precision_score(y,q),
        Brier=np.mean((y-q)**2),TP=tp,FP=fp,TN=tn,FN=fn,sensitivity=safe(tp,tp+fn),
        specificity=safe(tn,tn+fp),accuracy=(tp+tn)/len(y),PPV=safe(tp,tp+fp),
        NPV=safe(tn,tn+fn),F1=safe(2*tp,2*tp+fp+fn))
    for key,value in expected.items():
        np.testing.assert_allclose(performance[key],value,rtol=1e-12,atol=1e-12,equal_nan=True)
    for block,path in paths.items():
        assert b.core.sha(path.read_bytes())==freeze["external_input_SHA256"][block]
    np.testing.assert_array_equal(q,b.predict_final(payload,meta.assign(group=999,biologic="IGNORED",subtype=999,age=-999),species,pwy))
    result=dict(status="PASS",N=len(meta),predictions_reproduced=True,manual_transform_verified=True,
        metrics_reproduced=True,public_threshold_unchanged=True,model_hash_unchanged=True,
        clinical_and_outcome_columns_not_predictors=True)
    b.core.dump(output/"INDEPENDENT_EXTERNAL_VERIFICATION.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir",type=Path,default=Path("outputs/development"))
    parser.add_argument("--input-dir",type=Path,default=Path("data/external"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.development_dir,args.input_dir,args.output or args.development_dir/"external"),indent=2))
