"""Publication-oriented diagnostics from saved predictions; never fit models."""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve


COLORS={"association_F":"#2879A9","ElasticNet_nonzero_rank":"#A75A85","ExtraTrees_RFE":"#29876B"}


def save(fig,path):
    fig.tight_layout()
    for suffix in ("pdf","png"):
        fig.savefig(path.with_suffix("."+suffix),dpi=300,bbox_inches="tight")
    plt.close(fig)


def curves(frame,path,title):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    groups=frame.groupby("validation_group",sort=True) if "validation_group" in frame else [("Chinese",frame)]
    for name,part in groups:
        y=part.y_nonresponse.to_numpy()
        q=part.probability_nonresponse.to_numpy()
        if len(np.unique(y))<2:
            continue
        fpr,tpr,_=roc_curve(y,q)
        precision,recall,_=precision_recall_curve(y,q)
        axes[0].plot(fpr,tpr,label=f"{name}: AUC={auc(fpr,tpr):.3f}")
        axes[1].plot(recall,precision,label=f"{name}: AP={average_precision_score(y,q):.3f}")
    axes[0].plot([0,1],[0,1],"--",color="#999999",lw=1)
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8,loc="lower right")
    axes[0].set(xlabel="1 - Specificity",ylabel="Sensitivity",title=title+": ROC")
    axes[1].set(xlabel="Recall (Non-response)",ylabel="Precision (Non-response)",title=title+": PR")
    save(fig,path)


def development_figures(root,out):
    scores=pd.read_csv(root/"candidate_pipeline_scores.csv")
    valid=scores.loc[scores.status.eq("valid")].copy()
    winners=valid.loc[valid.groupby(["method","feature_count"]).mean_group_AUROC.idxmax()].copy()
    winners.to_csv(out/"best_configuration_per_selector_count.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4.7))
    for method,part in winners.groupby("method"):
        part=part.sort_values("mean_selected_features")
        ax.errorbar(part.mean_selected_features,part.mean_group_AUROC,yerr=part.SD_group_AUROC,
            marker="o",capsize=3,label=method,color=COLORS.get(method,"#6D7077"))
    ax.set(xlabel="Total selected features (mean across folds; log scale)",ylabel="Mean LOGO AUROC",
        xscale="log",ylim=(0,1),title="Development-optimized feature-count comparison")
    ax.legend(fontsize=8)
    save(fig,out/"feature_count_vs_LOGO_AUROC")
    hyper=valid.groupby("configuration_id").agg(best_mean_AUROC=("mean_group_AUROC","max"),
        tested_pipelines=("candidate_id","size")).reset_index().sort_values("best_mean_AUROC",ascending=False)
    params=valid[["configuration_id","parameters_json"]].drop_duplicates()
    hyper.merge(params,on="configuration_id",validate="one_to_one").to_csv(out/"hyperparameter_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,4.8))
    ax.bar(hyper.configuration_id,hyper.best_mean_AUROC,color="#2879A9")
    ax.tick_params(axis="x",rotation=90,labelsize=7)
    ax.set(ylabel="Best development mean LOGO AUROC",ylim=(0,1),title="Descriptive hyperparameter comparison")
    save(fig,out/"hyperparameter_comparison")
    predictions=pd.read_csv(root/"selected_pipeline_OOF_predictions.csv",float_precision="round_trip")
    curves(predictions,out/"group_specific_ROC_PR","Selected development pipeline")
    selected=pd.read_csv(root/"selected_pipeline_fold_features.csv")
    freq=pd.read_csv(root/"selected_pipeline_feature_stability.csv").sort_values(["fold_count","block","feature"],ascending=[False,True,True])
    columns=["G1","G2","G3"]
    presence=selected.assign(present=1).pivot(index="feature",columns="validation_group",values="present").fillna(0)
    presence=presence.reindex(index=freq.feature,columns=columns,fill_value=0)
    labels=[textwrap.fill(n.removeprefix("species__s__").removeprefix("pwy__"),64)+f" ({count}/3)"
        for n,count in zip(freq.feature,freq.fold_count)]
    fig,ax=plt.subplots(figsize=(12,max(4,len(labels)*.27)))
    ax.imshow(presence.to_numpy(),aspect="auto",vmin=0,vmax=1,cmap="Greens",interpolation="nearest")
    ax.set_xticks(range(3),columns)
    ax.set_yticks(range(len(labels)),labels,fontsize=7)
    ax.set_title("Training-fold feature selection stability (descriptive)")
    save(fig,out/"feature_stability_heatmap")
    features=pd.read_csv(root/"final/final_selected_features.csv")
    if "MDI_importance" in features:
        features=features.sort_values("MDI_importance")
        fig,ax=plt.subplots(figsize=(11,max(4,len(features)*.32)))
        ax.barh(range(len(features)),features.MDI_importance,color=features.block.map({"Species":"#29876B","PWY":"#B95F69"}))
        ax.set_yticks(range(len(features)),[textwrap.fill(n,70) for n in features.feature],fontsize=7)
        ax.set(xlabel="Final-model impurity importance (MDI, not PLS VIP)",title="Final full-public signature")
        save(fig,out/"final_feature_importance")
    plan=json.loads((root/"analysis_plan.json").read_text())
    pipeline=json.loads((root/"selected_pipeline.json").read_text())["candidate"]
    cutoff=json.loads((root/"locked_threshold.json").read_text())["threshold_nonresponse"]
    methods=(
        "METHODS\n\nWe developed a microbiome-only classifier in175 public patients using fixed three-group "
        "leave-one-group-out cross-validation (validation sizes114,39,22). For every candidate, raw prevalence>=20% and "
        "sample-variance>1e-12 filtering, Species natural-log CLR, and standardization were fitted only "
        "within the two training groups. The CLR pseudocount was half the minimum positive training "
        "abundance, and all training-retained Species formed the CLR reference. PWY remained on the "
        "original normalized scale, excluding UNMAPPED/UNINTEGRATED. Training means and sample SDs "
        "were applied unchanged to held-out patients. Species and PWY were selected jointly without "
        "per-block quotas using ANOVA-F, nonzero Elastic Net ranking or ExtraTrees-RFE. "
        +f"The predeclared run evaluated {plan['candidate_N']} pipeline configurations. "
        +plan["selection_rule"]+" "
        "The chosen procedure was refitted in all175 public patients. The maximum-Youden-J cutoff "
        +f"was locked at non-response probability>={cutoff:.17g} using public pooled OOF predictions. "
        "External data were not used for development or threshold optimization. Non-response was "
        "the positive event; AUPRC denotes average precision. The same LOGO folds were used for "
        "selection and evaluation; reported CV performance is internally optimized development "
        "performance, not unbiased nested-CV performance. The Chinese cohort was explored in earlier "
        "project iterations and is not prospectively untouched confirmatory validation.\n\n"
        "SELECTED CONFIGURATION\n"+json.dumps(pipeline,indent=2)+"\n")
    (out/"MANUSCRIPT_METHODS.txt").write_text(methods)


def external_figures(root,out):
    predictions=pd.read_csv(root/"external_predictions_with_outcomes.csv",float_precision="round_trip")
    curves(predictions,out/"external_ROC_PR","Locked Chinese evaluation")
    y=predictions.y_nonresponse.to_numpy()
    q=predictions.probability_nonresponse.to_numpy()
    cm=confusion_matrix(y,predictions.predicted_nonresponse,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4.5))
    ax.imshow(cm,cmap="Blues",vmin=0)
    for (row,col),value in np.ndenumerate(cm):
        ax.text(col,row,str(value),ha="center",va="center",fontsize=17)
    ax.set_xticks([0,1],["Response","Non-response"])
    ax.set_yticks([0,1],["Response","Non-response"])
    ax.set(xlabel="Predicted label",ylabel="True label",title="Non-response positive; locked public cutoff")
    save(fig,out/"external_confusion_matrix")
    true,predicted=calibration_curve(y,q,n_bins=4,strategy="quantile")
    pd.DataFrame(dict(mean_predicted_nonresponse=predicted,observed_nonresponse=true)).to_csv(out/"external_calibration_bins.csv",index=False)
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1],"--",color="#999999",label="Perfect calibration")
    ax.plot(predicted,true,"o-",color="#29876B",label="Original model probabilities")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted Non-response probability",
        ylabel="Observed Non-response proportion",title="Calibration assessment, 4 quantile bins")
    ax.legend(fontsize=8)
    save(fig,out/"external_calibration")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir",type=Path,default=Path("outputs/development"))
    parser.add_argument("--external-dir",type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs/figures"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"pdf.fonttype":42,"font.size":9})
    development_figures(args.development_dir,args.output)
    if args.external_dir:
        external_figures(args.external_dir,args.output)
    (args.output/"FIGURE_NOTES.txt").write_text(
        "No model fitting, selection or threshold change was performed. Feature-count and hyperparameter "
        "curves are maximized over development candidates, not unbiased validation estimates. Error bars "
        "show SD across the3 groups, not confidence intervals. External calibration bins are diagnostics "
        "only; no probability recalibration is applied.\n")


if __name__=="__main__":
    main()
