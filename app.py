#!/usr/bin/env python3
"""
Flask API for Attendance System with Face Recognition

Usage:
    python app.py

API Endpoints:
    POST /api/attendance
        - Upload an image file
        - Returns JSON with face recognition results
        - Content-Type: multipart/form-data
        - Form field: 'image' (file)

Example curl:
    curl -X POST -F "image=@classroom.jpg" http://localhost:5000/api/attendance
"""

import os
import sys
import io
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from infer_attendance import initialize_recognition_system, process_image_for_api

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables for model initialization (loaded once on startup)
embeddings_db = None
names_db = None
threshold = None

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """Initialize face recognition models using existing initialization function"""
    global embeddings_db, names_db, threshold
    
    embeddings_db, names_db, threshold = initialize_recognition_system(auto_calibrate=True)
    
    if embeddings_db is not None and names_db is not None and threshold is not None:
        print("[INFO] Models initialized successfully!")
        return True
    else:
        print("[ERROR] Failed to initialize models")
        return False

def process_image_data(image_data):
    """Process image data and return face recognition results using existing infer_attendance logic"""
    global embeddings_db, names_db, threshold
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Could not decode image data")
        
        results = process_image_for_api(
            img_bgr=img_bgr,
            embs_db=embeddings_db,
            names_db=names_db,
            threshold=threshold,
            ctx_id=-1,
            model_name="buffalo_l"
        )
        
        return results
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing image: {str(e)}",
            "total_faces": 0,
            "known_faces": 0,
            "unknown_faces": 0,
            "threshold_used": float(threshold) if threshold else 0.5,
            "recognized": []
        }

@app.route('/api/attendance', methods=['POST'])
def attendance_api():
    """Main API endpoint for face recognition"""
    
    if not all([embeddings_db is not None, names_db is not None, threshold]):
        return jsonify({
            "status": "error",
            "message": "Face recognition models not initialized"
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image file provided. Use 'image' as form field name."
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No image file selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        image_data = file.read()
        
        results = process_image_data(image_data)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_ready = all([embeddings_db is not None, names_db is not None, threshold])
    
    return jsonify({
        "status": "healthy" if models_ready else "initializing",
        "models_loaded": models_ready,
        "embeddings_count": len(embeddings_db) if embeddings_db is not None else 0,
        "unique_persons": len(set(names_db)) if names_db is not None else 0,
        "threshold": float(threshold) if threshold else None
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        "message": "Attendance System Face Recognition API",
        "version": "1.0",
        "endpoints": {
            "POST /api/attendance": "Upload image for face recognition",
            "GET /api/health": "Check API health and model status",
            "GET /": "This documentation"
        },
        "usage": "curl -X POST -F 'image=@your_image.jpg' http://localhost:5000/api/attendance"
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "status": "error",
        "message": f"File too large. Maximum size: {MAX_CONTENT_LENGTH // (1024*1024)}MB"
    }), 413

if __name__ == '__main__':
    print("Starting Attendance System Flask API...")
    
    if not initialize_models():
        print("[ERROR] Failed to initialize models. Exiting...")
        exit(1)
    
    print("[INFO] Starting Flask server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )