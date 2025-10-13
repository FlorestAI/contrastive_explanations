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