from typing import Dict, Tuple, List, Set, Any
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier

from pysat.formula import IDPool, WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc

# ----------------------------- utilitários comuns ----------------------------- #

def collect_thresholds(models: List[Any]) -> Dict[int, List[float]]:
    per_feat: Dict[int, Set[float]] = {} 
    for m in models: 
        estimators = [m] if hasattr(m, "tree_") else list(m.estimators_) 
        for est in estimators: 
            tr = est.tree_ 
            for f, t in zip(tr.feature, tr.threshold):
                if f != _tree.TREE_UNDEFINED: 
                    per_feat.setdefault(int(f), set()).add(float(t)) 
    return {j: sorted(list(vals)) for j, vals in per_feat.items()} 

def add_atleast_k(w: WCNF, lits: List[int], k: int, pool: IDPool): 
    if k <= 0: return # 
    enc = CardEnc.atleast(lits=lits, bound=k, vpool=pool, encoding=1)
    for cls in enc.clauses: w.append(cls)
    
def add_atmost_one(w: WCNF, lits: List[int]):
    # pairwise
    for i in range(len(lits)):
        for j in range(i+1, len(lits)):
            w.append([-lits[i], -lits[j]])
            
def add_sigma_monotonicity(w: WCNF, thresholds: Dict[int,List[float]], yvars: Dict[Tuple[int,float], int]):
    # y(j, t_high) -> y(j, t_low)  quando t_high > t_low
    for j, ts in thresholds.items():
        for i in range(len(ts)-1):
            t_low, t_high = ts[i], ts[i+1]
            w.append([-yvars[(j, t_high)], yvars[(j, t_low)]])
            
def add_soft_tx(w: WCNF, x: np.ndarray, thresholds: Dict[int,List[float]], yvars: Dict[Tuple[int,float], int]):
    # y_{j,t} := (x_j > t)  -> soft units
    for j, ts in thresholds.items():
        for t in ts:
            y = yvars[(j, t)]
            if x[j] > t:
                w.append([y], weight=1)
            else:
                w.append([-y], weight=1)
                
def diff_cost_from_model(model, yvars: Dict[Tuple[int,float], int], thresholds: Dict[int,List[float]], x: np.ndarray):
    if model is None: return None, []
    pos = set(l for l in model if l > 0)
    changes = []
    for j, ts in thresholds.items():
        for t in ts:
            v = yvars[(j, t)]
            want = 1 if x[j] > t else 0
            got  = 1 if (v in pos) else 0
            if got != want:
                changes.append((j, t, got))  # got=1 -> ">" ; got=0 -> "<="
    return len(changes), changes

def fmt_changes(changes, feature_names):
    out = []
    for j,t,got in sorted(changes, key=lambda z:(z[0], z[1])):
        sign = ">" if got==1 else "<="
        out.append(f"{feature_names[j]} {sign} {t:.3f}")
    return out