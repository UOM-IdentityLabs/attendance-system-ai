# infer_attendance.py
"""
Usage:
  python infer_attendance.py --image classroom.jpg --embeddings ./src/outputs/embeddings.pickle \
    --out ./output_annotated.jpg --threshold_file ./src/outputs/threshold.pickle --auto_calibrate 1

Notes:
 - Auto-calibration: if embeddings file mtime is newer than threshold file mtime, or threshold file missing,
   and auto_calibrate==1, the script will run a quick calibration automatically.
 - Manual override: pass --manual_threshold 0.7 to force using that similarity cutoff.
 - Output JSON is printed and saved to <out_json> (same structure you had).
"""
import os
import argparse
import pickle
import time
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_students_database(config_path="./src/config/students.json"):
    """
    Load students database from JSON configuration file
    
    Args:
        config_path: Path to the students configuration JSON file
        
    Returns:
        dict: Students database mapping
    """
    try:
        # Try multiple possible paths
        possible_paths = [
            config_path,
            "./config/students.json",
            os.path.join(os.path.dirname(__file__), "config", "students.json"),
            os.path.join(os.path.dirname(__file__), "..", "src", "config", "students.json")
        ]
        
        students_db = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                students_db = data.get("students", {})
                print(f"[INFO] Loaded students database from: {path}")
                break
        
            
        return students_db
        
    except Exception as e:
        print(f"[ERROR] Failed to load students database: {e}")
        

def load_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    embs = np.array(data["embeddings"])
    names = np.array(data["names"])
    return embs, names

def load_threshold(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("threshold")

def save_threshold(path, threshold, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"threshold": float(threshold), "meta": meta or {}}, f)

def calibrate_if_needed(emb_path, threshold_path, auto_calibrate):
    if not auto_calibrate:
        return load_threshold(threshold_path)
    emb_m = os.path.getmtime(emb_path)
    if not os.path.exists(threshold_path) or os.path.getmtime(threshold_path) < emb_m:
        print("[INFO] threshold missing or outdated, auto-calibrating...")
        # call the calibrator function inline for simplicity
        from calibrate_threshold import calibrate
        res = calibrate(emb_path, threshold_path)
        return res["threshold"]
    else:
        return load_threshold(threshold_path)

def initialize_recognition_system(embeddings_path=None, 
                                threshold_path=None,
                                auto_calibrate=True):
    """
    Initialize the complete face recognition system with embeddings and threshold
    
    Args:
        embeddings_path: Path to embeddings pickle file (auto-detected if None)
        threshold_path: Path to threshold pickle file (auto-detected if None)
        auto_calibrate: Whether to auto-calibrate threshold if needed
        
    Returns:
        tuple: (embeddings_db, names_db, threshold) or (None, None, None) on error
    """
    try:
        print("[INFO] Initializing face recognition system...")
        
        # Auto-detect paths if not provided
        if embeddings_path is None:
            possible_emb_paths = [
                "./outputs/embeddings.pickle",
                "./src/outputs/embeddings.pickle",
                os.path.join(os.path.dirname(__file__), "outputs", "embeddings.pickle")
            ]
            embeddings_path = next((p for p in possible_emb_paths if os.path.exists(p)), possible_emb_paths[0])
        
        if threshold_path is None:
            possible_thresh_paths = [
                "./outputs/threshold.pickle", 
                "./src/outputs/threshold.pickle",
                os.path.join(os.path.dirname(__file__), "outputs", "threshold.pickle")
            ]
            threshold_path = next((p for p in possible_thresh_paths if os.path.exists(p)), possible_thresh_paths[0])
        
        # Load embeddings
        embeddings_db, names_db = load_embeddings(embeddings_path)
        print(f"[INFO] Loaded {len(embeddings_db)} embeddings for {len(set(names_db))} unique persons")
        
        # Load or calibrate threshold
        threshold = calibrate_if_needed(embeddings_path, threshold_path, auto_calibrate)
        if threshold is None or threshold < 0.4:
            print("[WARN] Using conservative default threshold 0.5")
            threshold = 0.5
        else:
            print(f"[INFO] Using threshold: {threshold}")
        
        return embeddings_db, names_db, threshold
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize recognition system: {str(e)}")
        return None, None, None

def per_name_max_similarity(embs_db, names_db, query_emb):
    # normalize
    query = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    embs_norm = embs_db / (np.linalg.norm(embs_db, axis=1, keepdims=True) + 1e-12)
    sims = embs_norm @ query
    # compute per-name max
    per_name = {}
    for sim, name in zip(sims, names_db):
        if name not in per_name or sim > per_name[name]:
            per_name[name] = float(sim)
    # pick best
    best_name = max(per_name.items(), key=lambda kv: kv[1])
    return best_name  # (name, similarity)

def annotate_and_save(img_bgr, results, out_path):
    img = img_bgr.copy()
    for r in results:
        x, y, w, h = r["bbox"]
        if r["name"] == "unknown":
            color = (0, 0, 255)
            label = "unknown"
        else:
            color = (0, 255, 0)
            label = r["emp_id"] or r["name"]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        th = 1
        (tw, tht), baseline = cv2.getTextSize(label, font, fs, th)
        ly = max(y, tw+5)
        cv2.rectangle(img, (x, ly - tht - 5), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(img, label, (x+2, ly-2), font, fs, (0,0,0), th)
    cv2.imwrite(out_path, img)
    return out_path

def process_image_for_api(img_bgr, embs_db, names_db, threshold, ctx_id=-1, model_name="buffalo_l", students_config_path='./src/config/students.json'):
    """
    Process image for API usage - returns JSON-compatible results without saving files
    
    Args:
        img_bgr: OpenCV image in BGR format
        embs_db: Embeddings database
        names_db: Names corresponding to embeddings
        threshold: Similarity threshold for recognition
        ctx_id: Context ID for face analysis (-1 for CPU, 0+ for GPU)
        model_name: InsightFace model name
        students_config_path: Path to students JSON config file (optional)
    
    Returns:
        dict: JSON-compatible results
    """
    try:
        # Load dynamic students database
        students_db = load_students_database(students_config_path)
        
        
        # Initialize detector & embedder (FaceAnalysis)
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=ctx_id)

        # Convert to RGB for InsightFace
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        faces = app.get(img_rgb)
        results = []
        for f in faces:
            # bbox -> x1,y1,x2,y2 (float)
            x1, y1, x2, y2 = f.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            w = x2 - x1
            h = y2 - y1
            # embedding
            emb = np.array(f.embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)

            # per-name best similarity
            best_name, best_sim = per_name_max_similarity(embs_db, names_db, emb)

            # decide
            if best_sim >= threshold:
                # known - use dynamic students database
                student = students_db.get(best_name, {"name": best_name, "emp_id": None,})
                identified_name = student["name"]
                identified_emp_id = student["emp_id"]
                results.append({
                    "emp_id": identified_emp_id,
                    "name": identified_name,
                    "confidence": float(best_sim),
                    "distance": float(1.0 - best_sim),
                    "bbox": [int(x1), int(y1), int(w), int(h)]
                })
            else:
                results.append({
                    "emp_id": None,
                    "name": "unknown",
                    "confidence": float(best_sim),
                    "distance": float(1.0 - best_sim),
                    "bbox": [int(x1), int(y1), int(w), int(h)]
                })

        output = {
            "status": "success",
            "total_faces": len(results),
            "known_faces": len([r for r in results if r["name"] != "unknown"]),
            "unknown_faces": len([r for r in results if r["name"] == "unknown"]),
            "threshold_used": float(threshold),
            "students_loaded": len(students_db) if students_db else 0,
            "recognized": results
        }
        
        return output
        
    except Exception as e:
        print(f"[ERROR] Error in process_image_for_api: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing image: {str(e)}",
            "total_faces": 0,
            "known_faces": 0,
            "unknown_faces": 0,
            "threshold_used": float(threshold) if threshold else 0.5,
            "students_loaded": 0,
            "recognized": []
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="./output_annotated.jpg")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--threshold_file", default="./src/outputs/threshold.pickle")
    ap.add_argument("--auto_calibrate", type=int, default=1, help="1 = auto recalibrate when embeddings changed")
    ap.add_argument("--manual_threshold", type=float, default=None, help="If set, override automatic threshold")
    ap.add_argument("--ctx", type=int, default=0, help="GPU id (0) or -1 for CPU")
    ap.add_argument("--model", default="buffalo_l")
    args = ap.parse_args()

    # Load DB
    embs_db, names_db = load_embeddings(args.embeddings)
    if len(embs_db) == 0:
        raise RuntimeError("No embeddings in DB")

    # Determine threshold
    threshold = None
    if args.manual_threshold is not None:
        threshold = args.manual_threshold
        print(f"[INFO] Using manual threshold {threshold}")
    else:
        threshold = calibrate_if_needed(args.embeddings, args.threshold_file, auto_calibrate=bool(args.auto_calibrate))
        if threshold is None or threshold < 0.4:
            print("[WARN] No threshold found or threshold too low; using conservative default 0.5")
            threshold = 0.5
        else:
            print(f"[INFO] Using calculated threshold {threshold}")

    # Initialize detector & embedder (FaceAnalysis)
    app = FaceAnalysis(name=args.model)
    app.prepare(ctx_id=args.ctx)

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError("Could not read image: " + args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    faces = app.get(img_rgb)
    results = []
    for f in faces:
        # bbox -> x1,y1,x2,y2 (float)
        x1, y1, x2, y2 = f.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        w = x2 - x1
        h = y2 - y1
        # embedding
        emb = np.array(f.embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        # per-name best similarity
        best_name, best_sim = per_name_max_similarity(embs_db, names_db, emb)

        # decide
        if best_sim >= threshold:
            # known - use dynamic students database
            students_db = load_students_database()
            student = students_db.get(best_name, {"name": best_name, "emp_id": None})
            identified_name = student["name"]
            identified_emp_id = student["emp_id"]
            box_color = (0,255,0)
            results.append({
                "emp_id": identified_emp_id,
                "name": identified_name,
                "department": student.get("department"),
                "year": student.get("year"),
                "confidence": float(best_sim),
                "distance": float(1.0 - best_sim),  # for convenience keep a distance-like number
                "bbox": [int(x1), int(y1), int(w), int(h)]
            })
        else:
            results.append({
                "emp_id": None,
                "name": "unknown",
                "department": None,
                "year": None,
                "confidence": float(best_sim),
                "distance": float(1.0 - best_sim),
                "bbox": [int(x1), int(y1), int(w), int(h)]
            })

    output = {
        "status": "success",
        "total_faces": len(results),
        "known_faces": len([r for r in results if r["name"] != "unknown"]),
        "unknown_faces": len([r for r in results if r["name"] == "unknown"]),
        "recognized": results
    }

    # Annotate & save
    annotated_path = annotate_and_save(img_bgr, results, args.out)
    print(f"Annotated image saved: {annotated_path}")

    # Print & save JSON output (api-friendly)
    out_json = os.path.splitext(args.out)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
    print(f"JSON output saved: {out_json}")

if __name__ == "__main__":
    main()
