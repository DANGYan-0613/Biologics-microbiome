import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import logo_benchmark as b


class LOGOTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(23)
        ids = [f"s{i}" for i in range(60)]
        meta = pd.DataFrame(dict(id=ids, group=[0,1]*30,
            dataset=["PRJNA685168"]*20+["SRP057027"]*20+["PRJNA384246"]*20))
        s = pd.DataFrame(rng.uniform(.001,1,(60,5)),index=ids,columns=[f"species__s{i}" for i in range(5)])
        p = pd.DataFrame(rng.uniform(.001,1,(60,4)),index=ids,columns=[f"pwy__p{i}" for i in range(4)])
        return meta,s,p

    def test_grid_is_fixed_complete_product(self):
        grid = b.candidate_grid()
        self.assertEqual(len(grid),420)
        self.assertEqual(len({c["candidate_id"] for c in grid}),420)
        self.assertEqual(len(b.configuration_grid()),15)
        for m in b.METHODS:
            for k in b.COUNTS:
                self.assertEqual(sum(c["method"]==m and c["feature_count"]==k for c in grid),15)

    def test_no_heldout_in_preprocessing_or_any_selector(self):
        meta,s,p = self.fixture()
        train = meta.iloc[20:]
        state = b.core.fit_preprocessing(train,s,p)
        x = b.core.design(train,s,p,state)
        y = 1-train.group.to_numpy()
        before = {m:b.make_supports(x,y,m,counts=(2,4))[0] for m in b.METHODS}
        # Poison every held-out abundance, response and study label.
        s.iloc[:20] = 9999
        p.iloc[:20] = 0
        meta.loc[:19,"group"] = 1-meta.loc[:19,"group"]
        meta.loc[:19,"dataset"] = "POISONED"
        after = b.core.fit_preprocessing(meta.iloc[20:],s,p)
        self.assertEqual(state,after)
        changed = b.core.design(meta.iloc[20:],s,p,after)
        pd.testing.assert_frame_equal(x,changed)
        for m in b.METHODS:
            self.assertEqual(before[m],b.make_supports(changed,y,m,counts=(2,4))[0])

    def test_original_study_never_split(self):
        meta,_,_ = self.fixture()
        for _,tr,va in b.core.splits(meta):
            self.assertFalse(set(meta.id.iloc[tr]) & set(meta.id.iloc[va]))
            self.assertFalse(set(meta.dataset.iloc[tr]) & set(meta.dataset.iloc[va]))

    def test_rfe_has_exact_nested_counts_and_microbial_inputs_only(self):
        meta,s,p = self.fixture()
        x = pd.concat([s,p],axis=1)
        supports,_ = b.make_supports(x,1-meta.group.to_numpy(),"ExtraTrees_RFE",counts=(2,4,8))
        for k in (2,4,8):
            self.assertEqual(len(supports[k]),k)
        self.assertTrue(set(supports[2]) <= set(supports[4]) <= set(supports[8]))
        with self.assertRaisesRegex(ValueError,"microbial"):
            b.make_supports(x.assign(group=meta.group.to_numpy()),meta.group.to_numpy(),"association_F")

    def test_sparse_selector_never_pads_with_zero_coefficients(self):
        meta,s,p = self.fixture()
        x = pd.concat([s,p],axis=1)
        fake = type("EN",(),{"coef_":np.array([[0,2,0,0,0,0,0,0,-1.]])})()
        with patch.object(b.en,"fit_one",return_value=(fake,["species__s1","pwy__p3"],{"status":"valid"})):
            supports,_ = b.make_supports(x,meta.group.to_numpy(),"ElasticNet_nonzero_rank",counts=(1,2,4))
        self.assertEqual(len(supports[2]),2)
        self.assertNotIn(4,supports)

    def test_youden_nonresponse_direction_and_ties(self):
        y = np.array([0,0,1,1])
        q = np.array([.1,.4,.6,.9])
        threshold,_ = b.youden_threshold(y,q)
        self.assertEqual(threshold,.6)
        m = b.classification(y,q,threshold)
        self.assertEqual([m[k] for k in ("TP","FP","TN","FN")],[2,0,2,0])
        self.assertEqual(b.classification([1],[threshold],threshold)["TP"],1)

    def test_selector_parsimony_robustness_and_external_independence(self):
        scores = pd.DataFrame([
            dict(candidate_id="A",status="valid",mean_group_AUROC=.75,min_group_AUROC=.65,mean_selected_features=40,
                mean_pairwise_Jaccard=.3,pooled_nonresponse_AUPRC_AP=.7,depth_complexity=5,tree_complexity=700,leaf_complexity=2,C_complexity=0),
            dict(candidate_id="B",status="valid",mean_group_AUROC=.745,min_group_AUROC=.66,mean_selected_features=8,
                mean_pairwise_Jaccard=.5,pooled_nonresponse_AUPRC_AP=.69,depth_complexity=5,tree_complexity=700,leaf_complexity=2,C_complexity=0),
            dict(candidate_id="C",status="valid",mean_group_AUROC=.749,min_group_AUROC=.5,mean_selected_features=4,
                mean_pairwise_Jaccard=.9,pooled_nonresponse_AUPRC_AP=.8,depth_complexity=3,tree_complexity=700,leaf_complexity=2,C_complexity=0)])
        scores["Chinese_AUROC"] = [1,0,1]
        self.assertEqual(b.choose(scores).candidate_id,"B")

    def test_final_inference_ignores_metadata_values_other_than_ids(self):
        meta,s,p = self.fixture()
        state=b.core.fit_preprocessing(meta,s,p)
        x=b.core.design(meta,s,p,state)
        c=b.configuration_grid()[0]
        c={**c,"parameters":{**c["parameters"],"n_estimators":10}}
        model=b.fit_classifier(c,x,1-meta.group.to_numpy())
        payload=dict(preprocessing=state,features=x.columns.tolist(),classifier=model,target="Non-response = 1 - original group")
        before=b.predict_final(payload,meta,s,p)
        np.testing.assert_array_equal(before,b.predict_final(payload,meta.assign(group=9,dataset="ignored",subtype=999,biologic=999),s,p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
