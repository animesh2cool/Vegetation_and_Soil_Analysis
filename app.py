from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import base64
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import os
from typing import Optional, List, Dict
from pathlib import Path

app = FastAPI(
    title="VegeSegment API",
    description="AI-Powered Vegetation Segmentation & Soil Detection API",
    version="2.0.0"
)

# CORS Middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
vegetation_models = {}
soil_models = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320
N_CLASSES = 2  # Background and Vegetation

# Color map for vegetation visualization (RGB)
COLOR_MAP = {
    0: [0, 0, 0],           # Background - Black
    1: [34, 139, 34],       # Vegetation - Forest Green
}

def load_vegetation_models():
    """Load vegetation segmentation models"""
    print("🌿 Loading Vegetation Segmentation Models...")
    
    try:
        # Load U-Net
        if os.path.exists("best_unet.pth"):
            unet = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=N_CLASSES
            )
            unet.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
            unet.to(DEVICE)
            unet.eval()
            vegetation_models['unet'] = unet
            print("✅ U-Net loaded successfully")
        else:
            print("⚠️  U-Net model not found (best_unet.pth)")
        
        # Load DeepLabV3+
        if os.path.exists("best_deeplab.pth"):
            deeplab = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=N_CLASSES
            )
            deeplab.load_state_dict(torch.load("best_deeplab.pth", map_location=DEVICE))
            deeplab.to(DEVICE)
            deeplab.eval()
            vegetation_models['deeplab'] = deeplab
            print("✅ DeepLabV3+ loaded successfully")
        else:
            print("⚠️  DeepLabV3+ model not found (best_deeplab.pth)")
        
        # Load YOLO11-Seg
        yolo_path = "runs/segment/train/weights/best.pt"
        if os.path.exists(yolo_path):
            yolo = YOLO(yolo_path)
            vegetation_models['yolo'] = yolo
            print("✅ YOLO11-Seg loaded successfully")
        else:
            print(f"⚠️  YOLO11-Seg model not found ({yolo_path})")
        
        if not vegetation_models:
            print("❌ No vegetation models found!")
        else:
            print(f"🎉 Successfully loaded {len(vegetation_models)} vegetation model(s)")
        
    except Exception as e:
        print(f"❌ Error loading vegetation models: {e}")

def load_soil_models():
    """Load soil detection models"""
    print("\n🌍 Loading Soil Detection Models...")
    
    soil_model_files = {
        'yolo8': 'best_yolo8_soil.pt',
        'yolo11': 'best_yolo11_soil.pt',
        'rtdetr': 'best_RTDETR_soil.pt'
    }
    
    for model_name, model_file in soil_model_files.items():
        if os.path.exists(model_file):
            try:
                print(f"📦 Loading {model_name.upper()}...")
                model = YOLO(model_file)
                soil_models[model_name] = model
                print(f"✅ {model_name.upper()} loaded successfully")
            except Exception as e:
                print(f"⚠️  {model_name.upper()} failed to load: {str(e)[:100]}")
                print(f"   Skipping {model_name.upper()} due to compatibility issues")
        else:
            print(f"⚠️  {model_name.upper()} model not found ({model_file})")
    
    if not soil_models:
        print("❌ No soil detection models loaded!")
        print("   Expected files:")
        for name, file in soil_model_files.items():
            print(f"   - {file}")
    else:
        print(f"🎉 Successfully loaded {len(soil_models)} soil detection model(s)")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    print("="*60)
    print("🚀 Starting VegeSegment API Server")
    print(f"📍 Device: {DEVICE}")
    print("="*60)
    load_vegetation_models()
    load_soil_models()
    print("="*60)
    print("✅ Server ready!")
    print("="*60 + "\n")

def preprocess_image(image: Image.Image, size: int = IMG_SIZE):
    """Preprocess image for model input"""
    image = image.convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (size, size))
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor, image_np

def create_overlay(original_image, mask, alpha=0.5):
    """Create visualization overlay with colored vegetation"""
    h, w = original_image.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        colored_mask[mask_resized == class_id] = color
    
    # Blend with original
    overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    
    # Add contours for vegetation areas
    for class_id in range(1, N_CLASSES):
        contour_mask = (mask_resized == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    
    return overlay

def calculate_metrics(mask):
    """Calculate vegetation segmentation statistics"""
    total_pixels = mask.size
    vegetation_pixels = np.sum(mask > 0)
    vegetation_percentage = (vegetation_pixels / total_pixels) * 100
    
    return {
        "vegetation_percentage": round(vegetation_percentage, 2),
        "total_pixels": int(total_pixels),
        "vegetation_pixels": int(vegetation_pixels)
    }

def calculate_detection_stats(results) -> Dict:
    """Calculate statistics from YOLO detection results"""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "total_detections": 0,
            "unique_classes": 0,
            "avg_confidence": 0,
            "objects": []
        }
    
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    
    # Get class names
    names = results[0].names
    
    # Calculate statistics
    total_detections = len(boxes)
    unique_classes = len(set(classes))
    avg_confidence = round(float(np.mean(confidences) * 100), 2)
    
    # Create objects list
    objects = []
    for i in range(len(boxes)):
        class_id = int(classes[i])
        objects.append({
            "class": names[class_id],
            "confidence": round(float(confidences[i] * 100), 2),
            "class_id": class_id
        })
    
    return {
        "total_detections": total_detections,
        "unique_classes": unique_classes,
        "avg_confidence": avg_confidence,
        "objects": objects
    }

# ==================== ROOT & HEALTH ENDPOINTS ====================

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index.html", encoding="utf-8").read()

@app.get("/login", response_class=HTMLResponse)
async def login():
    return open("login.html", encoding="utf-8").read()

@app.get("/soil", response_class=HTMLResponse)
async def soil():
    return open("soil.html", encoding="utf-8").read()

@app.get("/vegetation", response_class=HTMLResponse)
async def vegetation():
    return open("vegetation.html", encoding="utf-8").read()


@app.get("/api")
async def root():
    """API root endpoint"""
    return {
        "message": "🌿 VegeSegment API - AI-Powered Analysis Platform",
        "version": "2.0.0",
        "modules": {
            "vegetation": {
                "available_models": list(vegetation_models.keys()),
                "total_models": len(vegetation_models)
            },
            "soil": {
                "available_models": list(soil_models.keys()),
                "total_models": len(soil_models)
            }
        },
        "device": str(DEVICE),
        "endpoints": {
            "vegetation_models": "/models",
            "vegetation_segment": "/segment/{model_name}",
            "soil_models": "/soil/models",
            "soil_detect": "/soil/detect/{model_name}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vegetation_models_loaded": len(vegetation_models),
        "soil_models_loaded": len(soil_models),
        "device": str(DEVICE)
    }

# ==================== VEGETATION SEGMENTATION ENDPOINTS ====================

@app.get("/models")
async def get_vegetation_models():
    """Get available vegetation segmentation models"""
    return {
        "models": [
            {
                "name": "unet",
                "display_name": "U-Net",
                "available": "unet" in vegetation_models,
                "description": "Medical imaging architecture with skip connections"
            },
            {
                "name": "deeplab",
                "display_name": "DeepLabV3+",
                "available": "deeplab" in vegetation_models,
                "description": "Multi-scale context with ASPP"
            },
            {
                "name": "yolo",
                "display_name": "YOLO11-Seg",
                "available": "yolo" in vegetation_models,
                "description": "Real-time detection and segmentation"
            }
        ],
        "device": str(DEVICE),
        "total_available": len(vegetation_models)
    }

@app.post("/segment/{model_name}")
async def segment_image(model_name: str, file: UploadFile = File(...)):
    """Perform vegetation segmentation on uploaded image"""
    
    if model_name not in vegetation_models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available: {list(vegetation_models.keys())}"
        )
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        print(f"📸 Processing vegetation with {model_name.upper()}: {file.filename}")
        
        # Process based on model type
        if model_name == 'yolo':
            results = vegetation_models['yolo'].predict(
                source=image, conf=0.25, retina_masks=True, verbose=False
            )
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                combined_mask = (masks.max(axis=0) > 0).astype(np.uint8)
            else:
                combined_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            image_tensor, original_image = preprocess_image(image)
            image_tensor = image_tensor.to(DEVICE)
            
            with torch.no_grad():
                output = vegetation_models[model_name](image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            result_img = create_overlay(original_image, pred_mask)
            combined_mask = pred_mask
        
        metrics = calculate_metrics(combined_mask)
        print(f"✅ Vegetation: {metrics['vegetation_percentage']}%")
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "model": model_name,
            "image": f"data:image/png;base64,{img_base64}",
            "metrics": metrics,
            "original_filename": file.filename
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SOIL DETECTION ENDPOINTS ====================

@app.get("/soil/models")
async def get_soil_models():
    """Get available soil detection models"""
    return {
        "models": [
            {
                "name": "yolo8",
                "display_name": "YOLO v8",
                "version": "v8",
                "available": "yolo8" in soil_models,
                "description": "Enhanced performance and speed"
            },
            {
                "name": "yolo11",
                "display_name": "YOLO v11",
                "version": "v11",
                "available": "yolo11" in soil_models,
                "description": "Latest YOLO with improved accuracy"
            },
            {
                "name": "rtdetr",
                "display_name": "RT-DETR",
                "version": "RT",
                "available": "rtdetr" in soil_models,
                "description": "Real-time transformer-based detector"
            }
        ],
        "device": str(DEVICE),
        "total_available": len(soil_models)
    }

@app.post("/soil/detect/{model_name}")
async def detect_soil_objects(
    model_name: str, 
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    """Perform soil object detection on uploaded image"""
    
    if model_name not in soil_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(soil_models.keys())}"
        )
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence threshold
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        print(f"🌍 Detecting soil with {model_name.upper()}: {file.filename}")
        print(f"   Confidence threshold: {conf}")
        
        # Run detection
        results = soil_models[model_name].predict(
            source=image,
            conf=conf,
            verbose=False
        )
        
        # Get annotated image
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Calculate statistics
        detections = calculate_detection_stats(results)
        
        print(f"✅ Detected {detections['total_detections']} objects")
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "model": model_name,
            "image": f"data:image/png;base64,{img_base64}",
            "detections": detections,
            "confidence_threshold": conf,
            "original_filename": file.filename
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")