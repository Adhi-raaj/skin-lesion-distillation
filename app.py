import os
import io
import logging
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from gradcam_service import GradCAMService
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel



def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Skin Lesion Classification API",
    description="Advanced AI-powered skin lesion detection and classification",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Class information
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus",
    "vasc": "Vascular Lesion"
}

# Model loading
try:
    # Try loading ONNX model
    model_path = os.getenv("MODEL_PATH", "skin_lesion_model.onnx")
    
    if not os.path.exists(model_path):
        # If ONNX model not found, provide helpful error message
        logger.warning(f"ONNX model not found at {model_path}")
        logger.info("To run this application:")
        logger.info("1. Convert your PyTorch model to ONNX: python convert_to_onnx.py")
        logger.info("2. Place the .onnx file in the project root or set MODEL_PATH environment variable")
        sess = None
    else:
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        logger.info(f"✓ ONNX Model loaded successfully: {model_path}")
        logger.info(f"  Input: {input_name}, Output: {output_name}")
        
except Exception as e:
    logger.error(f"Failed to load ONNX model: {str(e)}")
    sess = None

gradcam_service = None

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    success: bool

class ErrorResponse(BaseModel):
    error: str
    success: bool

# Image preprocessing
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed numpy array ready for inference
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0

        # ImageNet normalization (must match training)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_array = (img_array - mean) / std

        # Convert from HWC to CHW (Height-Width-Channel to Channel-Height-Width)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Skin Lesion Classification API",
        "model_loaded": sess is not None
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify a skin lesion image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction with confidence scores and probabilities for all classes
    """
    
    # Validate model is loaded
    if sess is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable. Please ensure the ONNX model is properly loaded."
        )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large (max 10MB)"
            )
        
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess image
        image_array = preprocess_image(contents)
        
        # Get input/output names
        input_name = sess.get_inputs()[0].name
        outputs = sess.run([output_name], {input_name: image_array})
        logits = outputs[0][0]

        # Convert logits to probabilities
        probs = softmax(logits)

        class_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 6)
            for i in range(len(CLASS_NAMES))
        }
        
        logger.info(
            f"Prediction: {predicted_class} "
            f"(confidence: {confidence:.4f}) "
            f"for {file.filename}"
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/gradcam")
async def generate_gradcam(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image_b64 = gradcam_service.generate(contents)

        return {
            "success": True,
            "gradcam_image": image_b64
        }

    except Exception as e:
        logger.exception("Grad-CAM generation failed")

        return {
            "success": False,
            "error": str(e)
        }

# Batch prediction endpoint (optional, for future use)
@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Classify multiple skin lesion images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of predictions
    """
    
    if sess is None:
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable"
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            
            if not contents or not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file"
                })
                continue
            
            image_array = preprocess_image(contents)
            input_name = sess.get_inputs()[0].name
            outputs = sess.run([output_name], {input_name: image_array})
            logits = outputs[0][0]

            probs = softmax(logits)

            class_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(probs[class_idx])

            probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 6)
            for i in range(len(CLASS_NAMES))
        }
            
            results.append({
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "success": True
            })
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

# Info endpoint
@app.get("/info")
async def get_info():
    """Get API information and available classes"""
    return {
        "service": "Skin Lesion Classification API",
        "version": "1.0.0",
        "classes": {class_name: CLASS_DESCRIPTIONS[class_name] for class_name in CLASS_NAMES},
        "model_input_size": 300,
        "model_input_shape": [1, 3, 300, 300],
        "expected_accuracy": 0.9062,
        "inference_latency_ms": 1.06
    }

# API documentation
@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Skin Lesion Classification API",
        "endpoints": {
            "health": "/health (GET) - Health check",
            "predict": "/predict (POST) - Single image prediction",
            "predict_batch": "/predict-batch (POST) - Batch predictions",
            "info": "/info (GET) - API information",
            "docs": "/docs - Interactive API documentation"
        },
        "usage": {
            "single_image": {
                "method": "POST",
                "endpoint": "/predict",
                "content_type": "multipart/form-data",
                "parameter": "file: <image_file>"
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("Skin Lesion Classification API Starting")
    logger.info("=" * 50)
    if sess is None:
        logger.warning("⚠️ Model not loaded! API will not function properly.")
        logger.info("Please ensure the ONNX model file exists.")
    else:
        logger.info("✓ All systems operational")

    global gradcam_service
    gradcam_service = GradCAMService()
    logger.info("✓ Grad-CAM model loaded successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False") == "True"
    
    logger.info(f"Starting server at {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
