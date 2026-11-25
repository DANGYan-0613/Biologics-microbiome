"""Develop a Species + PWY ExtraTrees prediction model using fixed LOGO-CV."""

import itertools
from pathlib import Path

import numpy as np

import logo_benchmark as benchmark

SPACE={"n_estimators":(300,700),"max_depth":(3,5,7),"min_samples_leaf":(2,4,6),
       "min_samples_split":(2,5),"max_features":("sqrt",.5),"class_weight":(None,"balanced")}


def configuration_grid():
    # Predeclared 24 base settings, each paired with both class-weight options.
    # Sampling is independent of all data/outcomes and precedes every model fit.
    base=list(itertools.product(SPACE["n_estimators"],SPACE["max_depth"],SPACE["min_samples_leaf"],
        SPACE["min_samples_split"],SPACE["max_features"]))
    anchors=list(itertools.product((700,),(3,5),(2,4),(2,),("sqrt",.5)))
    anchors.append((300,7,6,5,.5))
    remaining=[v for v in base if v not in anchors]
    rng=np.random.default_rng(benchmark.SEED)
    selected=anchors+[remaining[i] for i in rng.permutation(len(remaining))[:24-len(anchors)]]
    configurations=[]
    keys=list(SPACE)[:-1]
    for values in selected:
        for weight in SPACE["class_weight"]:
            params=dict(zip(keys,values))
            params.update(class_weight=weight,criterion="gini",bootstrap=False,random_state=benchmark.SEED,n_jobs=1)
            configurations.append(dict(configuration_id=f"ET{len(configurations):03d}",classifier="ExtraTrees",parameters=params))
    for name,values in SPACE.items():
        assert {c["parameters"][name] for c in configurations}==set(values)
    assert len(configurations)==48
    return configurations


if __name__=="__main__":
    benchmark.main(configuration_provider=configuration_grid,default_output=Path("outputs/development"))
