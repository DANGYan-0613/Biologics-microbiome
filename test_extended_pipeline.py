import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import logo_benchmark as b
import run_extratrees_logo as extended
import evaluate_external as evaluation
import validate_external_C0319 as source


class ExtendedTests(unittest.TestCase):
    def test_full_space_and_sample_coverage(self):
        configurations=extended.configuration_grid()
        self.assertEqual(np.prod([len(v) for v in extended.SPACE.values()]),144)
        self.assertEqual(len(configurations),48)
        self.assertEqual(len({json.dumps(c["parameters"],sort_keys=True) for c in configurations}),48)
        for name,values in extended.SPACE.items():
            self.assertEqual({c["parameters"][name] for c in configurations},set(values))
        self.assertTrue(all(c["classifier"]=="ExtraTrees" for c in configurations))
        self.assertEqual(len(b.candidate_grid(configurations)),1344)

    def test_paired_weights_and_reproducibility(self):
        configurations=extended.configuration_grid()
        self.assertEqual(configurations,extended.configuration_grid())
        for first,second in zip(configurations[::2],configurations[1::2]):
            a=first["parameters"].copy(); z=second["parameters"].copy()
            self.assertIsNone(a.pop("class_weight"))
            self.assertEqual(z.pop("class_weight"),"balanced")
            self.assertEqual(a,z)
            self.assertEqual(a["random_state"],20261125)
            self.assertFalse(a["bootstrap"])

    def test_bootstrap_does_not_fit_or_change_cutoff(self):
        y=np.array([0,0,0,1,1,1])
        q=np.array([.1,.2,.7,.3,.8,.9])
        with patch.object(b,"fit_classifier",side_effect=AssertionError("No fitting")), \
             patch.object(b,"youden_threshold",side_effect=AssertionError("No cutoff optimization")):
            intervals=evaluation.bootstrap_intervals(y,q,.6,repetitions=100)
        self.assertEqual(len(intervals),9)
        self.assertTrue(intervals.valid_replicates.gt(0).all())

    def test_ranking_diagnostics(self):
        x=pd.DataFrame({"species__a":[0,0,1,1],"pwy__b":[0,1,0,1]})
        support,diagnostic=b.make_supports(x,np.array([0,0,1,1]),"association_F",counts=(1,))
        self.assertEqual(support[1],["species__a"])
        self.assertEqual(diagnostic["training_ranking"][0]["feature"],"species__a")
        self.assertEqual(diagnostic["training_ranking"][0]["rank"],1)

    def test_end_to_end_synthetic_development_and_external(self):
        rng=np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inputs=root/"public"; inputs.mkdir()
            n=175
            ids=[f"PUBLIC_{i:03d}" for i in range(n)]
            meta=pd.DataFrame(dict(id=ids,group=[i%2 for i in range(n)],dataset=
                ["PRJNA685168"]*114+["SRP057027"]*39+["PRJNA384246"]*8+["PRJNA389280"]*7+["PRJNA398089"]*7))
            s=rng.uniform(.01,1,(n,8)); s/=s.sum(axis=1,keepdims=True)
            p=rng.uniform(.01,1,(n,6)); p/=p.sum(axis=1,keepdims=True)
            species=pd.DataFrame(s,index=ids,columns=[f"species__s{i}" for i in range(8)])
            pwy=pd.DataFrame(p,index=ids,columns=[f"pwy__p{i}" for i in range(6)])
            hashes={}
            for block,frame in (("meta",meta),("species",species.reset_index(names="id")),("pwy",pwy.reset_index(names="id"))):
                path=inputs/f"public_175_meta_species_pwy_raw_integrated_{block}.tsv"
                frame.to_csv(path,sep="\t",index=False)
                hashes[block]=b.core.sha(path.read_bytes())
            provider=lambda:[dict(configuration_id="SYNTHETIC",classifier="ExtraTrees",parameters=dict(
                n_estimators=8,max_depth=3,min_samples_leaf=2,max_features="sqrt",random_state=20261125,n_jobs=1))]
            loader=lambda directory,out:(meta,species,pwy)
            dev=root/"development"
            with patch.object(sys,"argv",["logo_benchmark.py","--input-dir",str(inputs),"--output",str(dev)]), \
                 patch.object(b.core,"EXPECTED_HASHES",hashes),patch.object(b.core,"load_public",loader), \
                 patch.object(b,"METHODS",("association_F",)),patch.object(b,"COUNTS",(2,4)):
                b.main(configuration_provider=provider)
            manifest=json.loads((dev/"analysis_plan.json").read_text())
            self.assertEqual(manifest["candidate_N"],3)
            q=pd.read_csv(dev/"selected_pipeline_OOF_predictions.csv")
            self.assertEqual(len(q),175)
            self.assertEqual(q.groupby("validation_group").size().to_dict(),{"G1":114,"G2":39,"G3":22})
            payload,_=evaluation.locked_payload(dev)
            self.assertIn(len(payload["features"]),(2,4,14))
            ext=root/"external_input"; ext.mkdir()
            eids=[f"SYNTHETIC_{i:02d}" for i in range(52)]
            emeta=pd.DataFrame(dict(id=eids,group=[0]*13+[1]*39))
            es=rng.uniform(.01,1,(52,8)); es/=es.sum(axis=1,keepdims=True)
            ep=rng.uniform(.01,1,(52,6)); ep/=ep.sum(axis=1,keepdims=True)
            hashes={}
            for block,frame in (("meta",emeta),("species",pd.DataFrame(es,columns=species.columns).assign(id=eids)),
                                ("pwy",pd.DataFrame(ep,columns=pwy.columns).assign(id=eids))):
                path=ext/f"yyyy_external_0928_pwy_matched_integrated_species_{block}.tsv"
                frame.to_csv(path,sep="\t",index=False)
                hashes[block]=b.core.sha(path.read_bytes())
            output=root/"external_output"
            with patch.object(sys,"argv",["evaluate_external.py","--development-dir",str(dev),"--input-dir",str(ext),
                "--output",str(output),"--bootstrap","100"]),patch.object(source,"EXTERNAL_SHA256",hashes), \
                patch.object(b,"fit_classifier",side_effect=AssertionError("No fitting in external evaluation")):
                evaluation.main()
            result=pd.read_csv(output/"external_performance.csv",float_precision="round_trip").iloc[0]
            self.assertEqual(result.N,52)
            self.assertEqual(result.TP+result.FN,13)
            self.assertEqual(result.TN+result.FP,39)
            self.assertEqual(result.threshold_nonresponse,payload["threshold_nonresponse"])
            import make_figures as figures
            rendered=[]
            def capture(fig,path):
                fig.canvas.draw()
                pixels=np.asarray(fig.canvas.buffer_rgba())
                self.assertGreater(pixels[...,:3].std(),1)
                rendered.append(path.name)
                figures.plt.close(fig)
            figure_output=root/"figures"; figure_output.mkdir()
            with patch.object(figures,"save",capture):
                figures.development_figures(dev,figure_output)
                figures.external_figures(output,figure_output)
            self.assertEqual(len(rendered),8)
            self.assertTrue((figure_output/"MANUSCRIPT_METHODS.txt").exists())


if __name__=="__main__":
    unittest.main(verbosity=2)
