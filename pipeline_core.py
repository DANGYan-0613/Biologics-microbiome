"""Exact public input, fold and preprocessing helpers from the completed run."""

import csv
import hashlib
import io
import json

import numpy as np
import pandas as pd

GROUPS = {"G1": ["PRJNA685168"], "G2": ["SRP057027"],
          "G3": ["PRJNA384246", "PRJNA389280", "PRJNA398089"]}
SEED = 20261125
NONBIOLOGICAL = {"pwy__UNMAPPED", "pwy__UNINTEGRATED"}
EXPECTED_HASHES = {"meta":"e68839b63d2b25cc206b34447af6f469f26886698409a199c89581efc8255251",
    "species":"94a88001d4338d4777b86a62b560b3f6d686ddd6689374a5b91d5bfc527b4676",
    "pwy":"c521a980c7dacefc3fa9f5abffad3e5aeeceb9a54bdce6766faf4643670af0b9"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")


def membership(meta):
    mapper={study:g for g,studies in GROUPS.items() for study in studies}
    groups=meta.dataset.map(mapper)
    if groups.isna().any(): raise ValueError("Unexpected public dataset")
    return groups


def splits(meta):
    groups=membership(meta).to_numpy()
    for group in GROUPS:
        if np.any(groups==group):
            tr,va=np.flatnonzero(groups!=group),np.flatnonzero(groups==group)
            if not len(tr): raise ValueError("At least two complete groups required")
            yield group,tr,va


def load_public(input_dir, out):
    frames,hashes,paths={},{},{}
    for block in ("meta","species","pwy"):
        path=input_dir/f"public_175_meta_species_pwy_raw_integrated_{block}.tsv"
        raw=path.read_bytes()
        header=next(csv.reader(io.StringIO(raw.decode("utf-8-sig")),delimiter="\t"))
        if len(header)!=len(set(header)): raise ValueError(f"Duplicate headers: {block}")
        hashes[block]=sha(raw); paths[block]=str(path)
        if hashes[block]!=EXPECTED_HASHES[block]: raise ValueError("Public input version changed; audit first")
        frames[block]=pd.read_csv(io.BytesIO(raw),sep="\t",dtype={"id":str})
        (out/"input_snapshot"/path.name).write_bytes(raw)
    meta=frames["meta"].sort_values("id").reset_index(drop=True)
    if len(meta)!=175 or meta.id.isna().any() or meta.id.duplicated().any() or set(meta.group)!={0,1}:
        raise ValueError("Exactly 175 uniquely identified binary-outcome patients required")
    if meta[["id","dataset","group"]].isna().any().any(): raise ValueError("Missing IDs/studies/outcomes")
    membership(meta)
    matrices={}
    for block in ("species","pwy"):
        frame=frames[block]
        if frame.id.isna().any() or frame.id.duplicated().any() or set(frame.id)!=set(meta.id):
            raise ValueError("IDs differ; no intersection-based deletion permitted")
        x=frame.set_index("id").loc[meta.id].astype(float)
        if not all(n.startswith(block+"__") for n in x) or not np.isfinite(x.to_numpy()).all() or (x<0).any().any():
            raise ValueError("Invalid abundance columns or values")
        matrices[block]=x
    if (matrices["species"].sum(axis=1)<=0).any(): raise ValueError("Empty Species profile")
    zeros=matrices["pwy"].index[matrices["pwy"].sum(axis=1).eq(0)].tolist()
    if set(zeros)!={"SRR2145528","SRR8956534","SRR8956568"}: raise ValueError("Zero-PWY sample set changed")
    matrices["pwy"]=matrices["pwy"].drop(columns=sorted(NONBIOLOGICAL))
    dump(out/"audit/data_alignment.json",dict(N=175,unique_ids=175,excluded_ids=[],retained_zero_PWY_ids=zeros,
        zero_PWY_policy="Stored raw zero encoding retained; not confirmed biological absence or imputed whole-block data",
        input_hashes=hashes,input_paths=paths,Species=matrices["species"].shape[1],PWY=matrices["pwy"].shape[1],
        metadata_predictors_used=False,Chinese_accessed=False))
    pd.concat([matrices["species"],matrices["pwy"]],axis=1).to_csv(out/"data/public_species_pwy_joint_raw.tsv",sep="\t",index_label="id")
    return meta,matrices["species"],matrices["pwy"]


def transform(raw,state):
    names=state["retained_raw_features"]
    if not set(names).issubset(raw.columns): raise ValueError("Missing retained features; no implicit zero filling")
    values=raw[names].copy()
    if state["representation"]=="clr":
        values=np.log(values+state["pseudocount"])
        values=values.sub(values.mean(axis=1),axis=0)
    return values


def fit_preprocessing(meta,species,pwy):
    state={}
    for block,raw,representation in (("Species",species,"clr"),("PWY",pwy,"original")):
        train=raw.loc[meta.id]
        names=train.columns[((train>0).mean(axis=0)>=.20)&(train.var(axis=0,ddof=1)>1e-12)].tolist()
        if not names: raise ValueError("No features survive training-only filtering")
        positives=train[names].to_numpy(); positives=positives[positives>0]
        current=dict(representation=representation,retained_raw_features=names,pseudocount=float(positives.min()/2),
                     prevalence_cutoff=.20,variance_epsilon=1e-12)
        values=transform(train,current)
        scale=values.std(axis=0,ddof=1).replace(0,1).replace([np.inf,-np.inf],1).fillna(1)
        current.update(center=values.mean(axis=0).tolist(),scale=scale.tolist())
        state[block]=current
    return state


def design(meta,species,pwy,state,names=None):
    blocks=[]
    for block,raw in (("Species",species),("PWY",pwy)):
        current=state[block]; columns=current["retained_raw_features"]
        values=transform(raw.loc[meta.id],current)
        values=(values-pd.Series(current["center"],index=columns))/pd.Series(current["scale"],index=columns)
        blocks.append(values)
    x=pd.concat(blocks,axis=1)
    if names is not None: x=x[names]
    if x.columns.duplicated().any() or not all(n.startswith(("species__","pwy__")) for n in x) or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Invalid microbial-only design")
    return x


def counts(names):
    return dict(Species=sum(n.startswith("species__") for n in names),PWY=sum(n.startswith("pwy__") for n in names))
