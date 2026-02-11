# calibrate_threshold.py
"""
Usage:
  python calibrate_threshold.py --embeddings ./src/outputs/embeddings.pickle --out ./src/outputs/threshold.pickle

This script computes an optimal cosine-similarity threshold by evaluating
same-person vs different-person similarities and choosing the threshold
that maximizes F1 score (youden or accuracy could be used too).
It saves threshold to a pickle: {"threshold": float, "meta": {...}}
"""
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import os

def load_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    embs = np.array(data["embeddings"])
    names = np.array(data["names"])
    return embs, names

def pairwise_similarities(embs, names):
    # compute pairwise cosine similarities
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim_matrix = embs_norm @ embs_norm.T
    n = embs.shape[0]
    sims = []
    labels = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(float(sim_matrix[i,j]))
            labels.append(1 if names[i]==names[j] else 0)
    return np.array(sims), np.array(labels)

def find_best_threshold(sims, labels, low=0.3, high=0.95, step=0.003):
    best = {"threshold": 0.5, "f1": -1}
    threshs = np.arange(low, high, step)
    for t in threshs:
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    return best

def calibrate(emb_path, out_path):
    embs, names = load_embeddings(emb_path)
    if len(embs) < 2:
        raise ValueError("Need at least 2 embeddings to calibrate threshold")
    sims, labels = pairwise_similarities(embs, names)
    best = find_best_threshold(sims, labels)
    meta = {
        "n_embeddings": len(embs),
        "unique_names": int(len(set(names))),
    }
    res = {"threshold": best["threshold"], "f1": best["f1"], "meta": meta}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"[INFO] saved threshold {res['threshold']} (f1={res['f1']}) to {out_path}")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", default="./src/outputs/threshold.pickle")
    args = ap.parse_args()
    calibrate(args.embeddings, args.out)
