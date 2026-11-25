"""External input alignment, coverage audit and locked transformation checks."""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

import logo_benchmark as b

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
