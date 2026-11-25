"""Independently recalculate saved LOGO results without training or selection fits."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import logo_benchmark as b


def close(a,c):
    np.testing.assert_allclose(a,c,rtol=1e-12,atol=1e-12)


def verify(out):
    scores=pd.read_csv(out/"candidate_pipeline_scores.csv",float_precision="round_trip")
    predictions=pd.read_csv(out/"candidate_OOF_predictions.csv",float_precision="round_trip")
    folds=pd.read_csv(out/"candidate_group_metrics.csv")
    plan=json.loads((out/"analysis_plan.json").read_text())
    selected=json.loads((out/"selected_pipeline.json").read_text())["candidate"]
    threshold=json.loads((out/"locked_threshold.json").read_text())["threshold_nonresponse"]
    audit=json.loads((out/"audit/data_alignment.json").read_text())
    meta=pd.read_csv(Path(audit["input_paths"]["meta"]),sep="\t").sort_values("id").reset_index(drop=True)
    species=pd.read_csv(Path(audit["input_paths"]["species"]),sep="\t").set_index("id").loc[meta.id]
    pwy=pd.read_csv(Path(audit["input_paths"]["pwy"]),sep="\t").set_index("id").loc[meta.id]
    pwy=pwy.drop(columns=["pwy__UNMAPPED","pwy__UNINTEGRATED"])
    for k,path in audit["input_paths"].items():
        assert b.core.sha(Path(path).read_bytes())==b.core.EXPECTED_HASHES[k]
        assert b.core.sha((out/"input_snapshot"/Path(path).name).read_bytes())==b.core.EXPECTED_HASHES[k]
    assert len(scores)==plan["candidate_N"]
    assert not plan["nested_CV"] and not plan["random_patient_splits"] and not plan["Chinese_access_allowed"]
    assert set(scores.candidate_id)=={c["candidate_id"] for c in plan["candidate_manifest"]}
    frozen={c["candidate_id"]:c for c in plan["candidate_manifest"]}
    for _,row in scores.iterrows():
        assert json.loads(row.parameters_json)==frozen[row.candidate_id]["parameters"]
    assert set(predictions.candidate_id)==set(scores.loc[scores.status.eq("valid"),"candidate_id"])
    assert set(folds.candidate_id)==set(predictions.candidate_id)
    assert not predictions.duplicated(["candidate_id","id"]).any()
    all_supports=pd.read_csv(out/"folds/all_training_selected_features.csv",dtype={"feature_count":str})
    support_lookup={}
    for (group,method,count),sub in all_supports.groupby(["validation_group","method","feature_count"]):
        assert not sub.feature.duplicated().any()
        state=json.loads((out/"folds"/group/"preprocessing.json").read_text())
        retained=set(state["Species"]["retained_raw_features"]+state["PWY"]["retained_raw_features"])
        assert set(sub.feature)<=retained
        if count!="ALL": assert len(sub)==int(count)
        else: assert set(sub.feature)==retained
        assert sub.block.eq(np.where(sub.feature.str.startswith("species__"),"Species","PWY")).all()
        support_lookup[group,method,count]=set(sub.feature)
    for identifier,p in predictions.groupby("candidate_id",sort=False):
        assert len(p)==175 and set(p.id)==set(meta.id)
        aligned=meta.set_index("id").loc[p.id]
        np.testing.assert_array_equal(p.y_nonresponse,1-aligned.group)
        np.testing.assert_array_equal(p.dataset,aligned.dataset)
        assert p.probability_nonresponse.between(0,1).all()
        row=scores.set_index("candidate_id").loc[identifier]
        close(roc_auc_score(p.y_nonresponse,p.probability_nonresponse),row.pooled_AUROC)
        close(average_precision_score(p.y_nonresponse,p.probability_nonresponse),row.pooled_nonresponse_AUPRC_AP)
        close(np.mean((p.y_nonresponse-p.probability_nonresponse)**2),row.pooled_Brier)
        aucs=[]
        feature_sets=[]
        for group,sub in p.groupby("validation_group"):
            assert len(sub)==b.EXPECTED_FOLDS[group][1]
            fr=folds.loc[folds.candidate_id.eq(identifier)&folds.validation_group.eq(group)].iloc[0]
            auc=roc_auc_score(sub.y_nonresponse,sub.probability_nonresponse)
            close(auc,fr.AUROC)
            close(average_precision_score(sub.y_nonresponse,sub.probability_nonresponse),fr.nonresponse_AUPRC_AP)
            wanted=support_lookup[group,row.method,str(row.feature_count)]
            assert fr.selected_features==len(wanted)
            assert fr.selected_species==sum(n.startswith("species__") for n in wanted)
            assert fr.selected_PWY==sum(n.startswith("pwy__") for n in wanted)
            feature_sets.append(wanted)
            aucs.append(auc)
        close(np.mean(aucs),row.mean_group_AUROC)
        close(min(aucs),row.min_group_AUROC)
        pairwise=[]
        for i in range(3):
            for j in range(i+1,3):
                pairwise.append(len(feature_sets[i]&feature_sets[j])/len(feature_sets[i]|feature_sets[j]))
        close(np.mean(pairwise),row.mean_pairwise_Jaccard)
        union=set.union(*feature_sets)
        for count in (1,2,3):
            assert row[f"features_{count}_of_3"]==sum(sum(n in s for s in feature_sets)==count for n in union)
    valid=scores.loc[scores.status.eq("valid")]
    eligible=valid.loc[valid.mean_group_AUROC>=valid.mean_group_AUROC.max()-.01]
    eligible=eligible.loc[eligible.min_group_AUROC>=eligible.min_group_AUROC.max()-.02]
    ordered=eligible.sort_values(["mean_selected_features","mean_pairwise_Jaccard","pooled_nonresponse_AUPRC_AP",
        "mean_group_AUROC","depth_complexity","tree_complexity","leaf_complexity","C_complexity","candidate_id"],
        ascending=[True,False,False,False,True,True,False,True,True])
    assert selected["candidate_id"]==ordered.iloc[0].candidate_id
    assert scores.loc[scores.selected,"candidate_id"].tolist()==[selected["candidate_id"]]
    chosen=predictions.loc[predictions.candidate_id.eq(selected["candidate_id"])].reset_index(drop=True)
    # Independent exhaustive Youden calculation at every unique saved OOF probability.
    trials=[]
    yy=chosen.y_nonresponse.to_numpy()
    qq=chosen.probability_nonresponse.to_numpy()
    for cutoff in sorted(np.unique(qq),reverse=True):
        pred=qq>=cutoff
        sensitivity=pred[yy==1].mean()
        specificity=(~pred[yy==0]).mean()
        trials.append((float(sensitivity+specificity-1),float(cutoff)))
    max_j=max(j for j,_ in trials)
    best_cutoffs=[c for j,c in trials if np.isclose(j,max_j,atol=1e-14,rtol=0)]
    close(threshold,max(best_cutoffs))
    performance=pd.read_csv(out/"selected_pipeline_development_performance.csv")
    for _,row in performance.iterrows():
        sub=chosen if row.evaluation=="Pooled" else chosen.loc[chosen.validation_group.eq(row.evaluation)]
        pred=sub.probability_nonresponse>=threshold
        tn,fp,fn,tp=confusion_matrix(sub.y_nonresponse,pred,labels=[0,1]).ravel()
        np.testing.assert_array_equal([row.TP,row.FP,row.TN,row.FN],[tp,fp,tn,fn])
        close(row.sensitivity,tp/(tp+fn))
        close(row.specificity,tn/(tn+fp))
        close(row.accuracy,(tn+tp)/len(sub))
        if tp+fp: close(row.PPV,tp/(tp+fp))
        if tn+fn: close(row.NPV,tn/(tn+fn))
        if 2*tp+fp+fn: close(row.F1,2*tp/(2*tp+fp+fn))
    selected_features=pd.read_csv(out/"selected_pipeline_fold_features.csv")
    stability=pd.read_csv(out/"selected_pipeline_feature_stability.csv")
    assert not selected_features.duplicated(["validation_group","feature"]).any()
    frequency=selected_features.groupby("feature").validation_group.nunique()
    for _,row in stability.iterrows():
        assert row.fold_count==frequency[row.feature]
        close(row.frequency,row.fold_count/3)
    summary=[]
    for block in ("Species","PWY"):
        for count in (3,2,1):
            summary.append(dict(block=block,selected_in_folds=f"{count}/3",feature_N=int(
                (stability.block.eq(block)&stability.fold_count.eq(count)).sum())))
    pd.DataFrame(summary).to_csv(out/"selected_pipeline_stability_complete_summary.csv",index=False)
    for group,tr,va in b.core.splits(meta):
        model=joblib.load(out/"folds"/group/"selected_development_fold_model.joblib")
        provenance=json.loads((out/"folds"/group/"training_provenance.json").read_text())
        assert provenance["train_ids"]==meta.id.iloc[tr].tolist()
        assert provenance["validation_ids"]==meta.id.iloc[va].tolist()
        assert not set(meta.dataset.iloc[tr])&set(meta.dataset.iloc[va])
        np.testing.assert_array_equal(model["features"],selected_features.loc[selected_features.validation_group.eq(group),"feature"])
        expected=chosen.set_index("id").loc[meta.id.iloc[va],"probability_nonresponse"]
        close(b.predict_final(model,meta.iloc[va],species,pwy),expected)
        # Validate fold-specific retained features and parameters independently with numpy.
        for block,raw in (("Species",species),("PWY",pwy)):
            a=raw.loc[meta.id.iloc[tr]].to_numpy(float)
            keep=((a>0).mean(axis=0)>=.20)&(a.var(axis=0,ddof=1)>1e-12)
            state=model["preprocessing"][block]
            assert raw.columns[keep].tolist()==state["retained_raw_features"]
            a=a[:,keep]
            if block=="Species":
                pseudo=a[a>0].min()/2
                close(pseudo,state["pseudocount"])
                a=np.log(a+pseudo)
                a-=a.mean(axis=1,keepdims=True)
            close(a.mean(axis=0),state["center"])
            sd=a.std(axis=0,ddof=1)
            sd[(sd==0)|~np.isfinite(sd)]=1
            close(sd,state["scale"])
    final=joblib.load(out/"final/Microbiome_LOGO_selected_final_175.joblib")
    matrix=b.core.design(meta,species,pwy,final["preprocessing"],final["features"])
    saved=pd.read_csv(out/"final/final_training_matrix.tsv",sep="\t",index_col="id")
    close(matrix,saved)
    registry=json.loads((out/"final/model_registry.json").read_text())
    assert registry["model_SHA256"]==b.core.sha((out/"final/Microbiome_LOGO_selected_final_175.joblib").read_bytes())
    assert final["classifier"].get_params()==registry["parameters"]
    close(final["threshold_nonresponse"],threshold)
    poisoned=meta.assign(group=9,dataset="IGNORED",subtype="IGNORED",biologic="IGNORED",adult="IGNORED")
    np.testing.assert_array_equal(b.predict_final(final,meta,species,pwy),b.predict_final(final,poisoned,species,pwy))
    for source in (Path(b.__file__),Path(b.core.__file__),Path(b.en.__file__)):
        assert b.core.sha(source.read_bytes())==b.core.sha((out/"code_snapshot"/source.name).read_bytes())
    result=dict(status="PASS",candidate_N=len(scores),valid_candidate_N=int(scores.status.eq("valid").sum()),
        OOF_rows=len(predictions),all_candidate_scores_reproduced=True,all_selected_fold_predictions_reproduced=True,
        training_only_preprocessing_independently_verified=True,feature_frequency_verified=True,
        full175_training_matrix_and_serialization_verified=True,public_Youden_threshold_verified=True,
        selection_rule_reproduced=True,metadata_predictor_invariance_verified=True,input_and_code_hashes_verified=True,
        model_training_during_verification=False,Chinese_accessed=False,interpretation=b.LIMITATION)
    b.core.dump(out/"INDEPENDENT_VERIFICATION.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/development"))
    verify(parser.parse_args().output)
