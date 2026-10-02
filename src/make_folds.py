"""Canonical lesion-level stratified 5-fold split (seed 42). Used by all later steps.
Note: the metadata has no patient ID, so lesions of the same patient can fall into different folds."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import ROOT, CLASSES

gt = pd.read_csv(ROOT / 'MILK10k_Training_GroundTruth.csv')
y = gt[CLASSES].values.argmax(1)
gt['fold'] = -1
for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(gt, y)):
    gt.loc[te, 'fold'] = f
gt['dx'] = [CLASSES[i] for i in y]
gt[['lesion_id', 'fold', 'dx']].to_csv(ROOT / 'data' / 'folds.csv', index=False)
print(pd.crosstab(gt['dx'], gt['fold']))
