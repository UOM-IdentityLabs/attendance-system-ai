# create_embeddings.py
"""
Usage:
  python create_embeddings.py --dataset ./src/dataset/train --out ./src/outputs/embeddings.pickle --ctx 0

This will:
 - Walk dataset where folder names are labels (same as your current layout)
 - Use InsightFace FaceAnalysis (RetinaFace + ArcFace) to detect and extract embeddings
 - Save a pickle: {"embeddings": [ndarray,...], "names": [str,...]}
"""
import os
import argparse
import pickle
import numpy as np
from imutils import paths
from insightface.app import FaceAnalysis
import cv2

def build_embeddings(dataset_dir, out_path, ctx_id=0, model_name="buffalo_l"):
    app = FaceAnalysis(name=model_name)     # "buffalo_l" is accurate and stable; "antelopev2" is faster with similar perf
    app.prepare(ctx_id=ctx_id)              # ctx_id=0 for GPU, -1 for CPU

    imagePaths = list(paths.list_images(dataset_dir))
    embeddings = []
    names = []
    total = 0

    for i, imagePath in enumerate(imagePaths, start=1):
        print(f"[INFO] ({i}/{len(imagePaths)}) {imagePath}")
        name = os.path.basename(os.path.dirname(imagePath))
        img = cv2.imread(imagePath)
        if img is None:
            print("[WARN] could not read:", imagePath)
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = app.get(img_rgb)
        if len(faces) == 0:
            print("[WARN] no face detected, skipping:", imagePath)
            continue

        # Choose the largest face if multiple (typical for training images)
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.embedding  # already normalized in recent insightface versions (but we normalize again to be safe)
        emb = np.asarray(emb, dtype=np.float32)
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        embeddings.append(emb)
        names.append(name)
        total += 1

    print(f"[INFO] {total} faces embedded")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names}, f)
    print(f"[INFO] embeddings saved to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="./src/outputs/embeddings.pickle")
    ap.add_argument("--ctx", type=int, default=0, help="GPU id (0) or -1 for CPU")
    ap.add_argument("--model", default="buffalo_l", help="InsightFace model name (buffalo_l / antelopev2)")
    args = ap.parse_args()

    build_embeddings(args.dataset, args.out, ctx_id=args.ctx, model_name=args.model)
