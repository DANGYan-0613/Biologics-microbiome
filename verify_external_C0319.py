"""Read-only external-result verification; no fitting, selection or threshold tuning."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import validate_external_C0319 as v


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir",type=Path,default=v.DEVELOPMENT)
    parser.add_argument("--input-dir",type=Path,default=v.INPUT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    out=args.output or args.development_dir/"external_chinese_01"
    payload,model_path=v.locked_model(args.development_dir)
    (meta,species,pwy),paths=v.load_tables(args.input_dir)
    predictions=pd.read_csv(out/"external_predictions_before_outcome_join.csv",float_precision="round_trip")
    evaluated=pd.read_csv(out/"external_predictions_with_outcomes.csv",float_precision="round_trip")
    performance=pd.read_csv(out/"external_performance.csv",float_precision="round_trip").iloc[0]
    freeze=json.loads((out/"prediction_freeze.json").read_text())
    assert len(predictions)==52 and predictions.id.nunique()==52
    assert predictions.id.tolist()==meta.id.tolist()==evaluated.id.tolist()
    assert not {"group","original_group","y_nonresponse","age","subtype","biologic"}&set(predictions.columns)
    assert freeze["SHA256"]==v.b.core.sha((out/"external_predictions_before_outcome_join.csv").read_bytes())
    for block,path in paths.items():
        assert v.b.core.sha(path.read_bytes())==v.EXTERNAL_SHA256[block]
        assert v.b.core.sha((out/"input_snapshot"/path.name).read_bytes())==v.EXTERNAL_SHA256[block]
    assert v.b.core.sha(model_path.read_bytes())==v.MODEL_SHA256
    assert v.b.core.sha(Path(v.__file__).read_bytes())==v.b.core.sha((out/"validate_external_C0319_frozen.py").read_bytes())
    assert freeze["model_SHA256"]==v.MODEL_SHA256
    for key in ("training","feature_selection","tuning","calibration","threshold_optimization","outcome_or_clinical_predictors_used"):
        assert freeze[key] is False
    q=predictions.probability_nonresponse.to_numpy()
    y=1-meta.group.to_numpy(dtype=int)
    cutoff=json.loads((args.development_dir/"locked_threshold.json").read_text())["threshold_nonresponse"]
    assert cutoff==payload["threshold_nonresponse"]==freeze["threshold_nonresponse"]
    np.testing.assert_array_equal(predictions.threshold_nonresponse,np.full(52,cutoff))
    np.testing.assert_array_equal(q,evaluated.probability_nonresponse)
    np.testing.assert_array_equal(1-q,predictions.probability_response)
    np.testing.assert_array_equal(evaluated.y_nonresponse,y)
    pred=q>=cutoff
    np.testing.assert_array_equal(pred,predictions.predicted_nonresponse)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    expected=dict(AUROC=roc_auc_score(y,q),nonresponse_AUPRC_AP=average_precision_score(y,q),
        Brier=np.mean((y-q)**2),TP=tp,FP=fp,TN=tn,FN=fn,sensitivity=tp/(tp+fn),
        specificity=tn/(tn+fp),accuracy=(tp+tn)/52,PPV=tp/(tp+fp),NPV=tn/(tn+fn),F1=2*tp/(2*tp+fp+fn))
    for key,value in expected.items():
        np.testing.assert_allclose(performance[key],value,atol=1e-14,rtol=1e-14)
    raw_model=payload["classifier"]
    manual=v.independent_design(payload,species,pwy)
    np.testing.assert_array_equal(q,raw_model.predict_proba(manual.to_numpy())[:,1])
    saved=pd.read_csv(out/"external_locked_transformed_52x16.tsv",sep="\t",index_col="id",float_precision="round_trip")
    assert saved.columns.tolist()==payload["features"]
    np.testing.assert_allclose(saved.to_numpy(),manual.to_numpy(),atol=1e-12,rtol=1e-12)
    # Explicitly forbid fitting/selection routines while testing the inference entry point.
    with patch.object(v.b,"fit_classifier",side_effect=AssertionError("No fitting allowed")), \
         patch.object(v.b,"make_supports",side_effect=AssertionError("No selection allowed")), \
         patch.object(v.b.core,"fit_preprocessing",side_effect=AssertionError("No preprocessing fitting allowed")):
        actual=v.b.predict_final(payload,meta[["id"]],species,pwy)
        np.testing.assert_array_equal(actual,q)
        changed=meta.assign(group=9,subtype=999,biologic="IGNORED",age=999,dataset="IGNORED")
        np.testing.assert_array_equal(q,v.b.predict_final(payload,changed,species,pwy))
        np.testing.assert_array_equal(q,v.b.predict_final(payload,meta[["id"]],species[species.columns[::-1]],pwy[pwy.columns[::-1]]))
    absent=next(n for n in payload["preprocessing"]["Species"]["retained_raw_features"] if n not in payload["features"])
    try:
        v.b.predict_final(payload,meta[["id"]],species.drop(columns=absent),pwy)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing unselected CLR component should stop inference")
    frames={"meta":meta,"species":species.reset_index(),"pwy":pwy.reset_index()}
    malformed=dict(frames,species=pd.concat([frames["species"],frames["species"].iloc[:1]],ignore_index=True))
    try:
        v.align_tables(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate external sample IDs must stop inference")
    coverage=pd.read_csv(out/"all_required_feature_coverage_and_raw_distribution.csv")
    selected=pd.read_csv(out/"selected_16_feature_audit.csv")
    assert len(coverage)==507 and coverage.present_in_external.all() and not coverage.imputed_zero.any()
    assert selected.feature.tolist()==payload["features"] and len(selected)==16
    assert selected.varies_across_external.all()
    cm=pd.read_csv(out/"confusion_matrix_nonresponse_positive.csv",index_col="true_label")
    np.testing.assert_array_equal(cm.to_numpy(),[[tn,fp],[fn,tp]])
    result=dict(status="PASS",N=52,nonresponse_N=13,response_N=39,
        model_SHA256_unchanged=True,external_input_SHA256_unchanged=True,
        frozen_label_free_predictions_reproduced=True,manual_preprocessing_verified=True,
        all_performance_and_confusion_counts_reproduced=True,
        public_threshold_unchanged=True,clinical_and_outcome_columns_not_predictors=True,
        no_training_selection_or_tuning=True,missing_CLR_component_rejected=True,
        duplicate_ids_rejected=True,required_components_present=507,imputed_columns=0,
        selected_predictors_varying=16,calibrated=False)
    v.b.core.dump(out/"INDEPENDENT_EXTERNAL_VERIFICATION.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
