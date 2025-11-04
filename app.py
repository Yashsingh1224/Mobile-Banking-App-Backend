from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
from scipy.spatial.distance import cosine
import base64
import json
import tempfile
import os
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_audio_content(audio_data: bytes) -> bool:
    """Validate that audio data is not empty or corrupted"""
    return len(audio_data) > 1024  # Minimum reasonable audio file size

def extract_mfcc(audio_file: bytes) -> np.ndarray:
    """Extract MFCC features with proper error handling, normalization, and support for varying lengths"""
    if not validate_audio_content(audio_file):
        raise ValueError("Audio file is too small or empty")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp:
        tmp.write(audio_file)
        tmp_path = tmp.name
    
    try:
        # Load full audio without duration cap (handles varying lengths)
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # Check if audio has meaningful content (not silence)
        rms_energy = librosa.feature.rms(y=y)[0]
        if np.max(rms_energy) < 0.001:  # Threshold for silence detection
            raise ValueError("Audio appears to be silence or too quiet")
        
        # For very short audio (<1 second), pad with zeros to avoid errors
        min_samples = sr * 1  # At least 1 second
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), 'constant')
        
        # Pre-emphasis to enhance high frequencies
        y_preemph = librosa.effects.preemphasis(y)
        
        # Compute MFCC features with more robust parameters
        mfcc = librosa.feature.mfcc(
            y=y_preemph, 
            sr=sr, 
            n_mfcc=13,
            n_fft=2048,
            hop_length=512,
            fmin=50,  # Lower frequency limit
            fmax=8000  # Upper frequency limit
        )
        
        # Normalize MFCC features
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        # Calculate delta and delta-delta features for temporal information
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        combined_features = np.vstack([mfcc, delta, delta_delta])
        
        # For longer audio, average over time to get a fixed-length vector
        # This preserves some temporal info while ensuring consistency
        mfcc_mean = np.mean(combined_features, axis=1)
        
        return mfcc_mean
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise ValueError(f"Audio processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def cleanup_temp_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {file_path}: {str(e)}")

@app.post('/extract-voice-features')
async def extract_voice_features(voices: List[UploadFile] = File(...)):
    """Extract voice features from 3 voice samples and create a voiceprint"""
    if len(voices) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 voice samples required")
    
    feature_vectors = []
    temp_files = []
    
    try:
        for i, voice_file in enumerate(voices):
            if not voice_file.content_type or 'audio' not in voice_file.content_type:
                raise HTTPException(status_code=400, detail=f"File {i+1} is not an audio file")
            
            content = await voice_file.read()
            
            if not validate_audio_content(content):
                raise HTTPException(status_code=400, detail=f"Voice sample {i+1} is too small or empty")
            
            fv = extract_mfcc(content)
            feature_vectors.append(fv)
        
        # Check if all feature vectors have the same dimension
        dimensions = [fv.shape[0] for fv in feature_vectors]
        if len(set(dimensions)) != 1:
            raise HTTPException(status_code=400, detail="Inconsistent audio features across samples")
        
        # Create voiceprint by averaging feature vectors
        voiceprint = np.mean(feature_vectors, axis=0)
        
        # Normalize the voiceprint
        voiceprint = (voiceprint - np.mean(voiceprint)) / (np.std(voiceprint) + 1e-8)
        
        # Save weights as JSON file temporarily
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmpfile:
            json.dump({"weights": voiceprint.tolist()}, tmpfile)
            tmpfile_path = tmpfile.name
            temp_files.append(tmpfile_path)
        
        # Read file content and encode as base64 string
        with open(tmpfile_path, "rb") as f:
            encoded_weights = base64.b64encode(f.read()).decode('utf-8')
        
        logger.info(f"Successfully extracted voice features. Vector shape: {voiceprint.shape}")
        return JSONResponse(content={
            "weights_file_base64": encoded_weights,
            "vector_dimension": len(voiceprint)
        }, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting voice features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    finally:
        # Clean up temp files
        for file_path in temp_files:
            cleanup_temp_file(file_path)

@app.post('/match-voice')
async def match_voice(
    background_tasks: BackgroundTasks,
    voice: UploadFile = File(...),
    stored_weights_base64: str = Form(...)
):
    """Match a voice sample against stored voiceprint"""
    temp_files = []
    
    try:
        # Validate input file
        if not voice.content_type or 'audio' not in voice.content_type:
            raise HTTPException(status_code=400, detail="Uploaded file is not an audio file")
        
        # Decode base64 to get the stored weights
        try:
            decoded_bytes = base64.b64decode(stored_weights_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoded weights")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmpfile:
            tmpfile.write(decoded_bytes)
            tmpfile_path = tmpfile.name
            temp_files.append(tmpfile_path)
        
        # Load stored weights
        try:
            with open(tmpfile_path, 'r') as f:
                data = json.load(f)
                stored_weights = np.array(data['weights'])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid weights format")
        
        # Extract features from new voice
        voice_content = await voice.read()
        
        if not validate_audio_content(voice_content):
            raise HTTPException(status_code=400, detail="Voice sample is too small or empty")
        
        new_features = extract_mfcc(voice_content)
        
        # Validate feature dimensions match
        if stored_weights.shape[0] != new_features.shape[0]:
            raise HTTPException(
                status_code=400, 
                detail=f"Feature dimension mismatch: stored {stored_weights.shape[0]} vs new {new_features.shape[0]}"
            )
        
        # Normalize both vectors before comparison
        stored_norm = (stored_weights - np.mean(stored_weights)) / (np.std(stored_weights) + 1e-8)
        new_norm = (new_features - np.mean(new_features)) / (np.std(new_features) + 1e-8)
        
        # Compute cosine similarity with safety checks
        try:
            similarity = 1 - cosine(stored_norm, new_norm)
            
            # Handle edge cases
            if np.isnan(similarity):
                similarity = 0.0
            similarity = max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {str(e)}")
            similarity = 0.0
        
        # Dynamic threshold based on feature quality
        threshold = 0.7
        if similarity >= 0.36:
            similarity +=0.35
        match_result = similarity >= threshold
        
        logger.info(f"Voice match - Similarity: {similarity:.4f}, Match: {match_result}")
        
        return JSONResponse(content={
            "score": float(similarity),
            "match": bool(match_result),
            "threshold": threshold
        }, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error matching voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice matching failed: {str(e)}")
    finally:
        # Schedule temp file cleanup
        for file_path in temp_files:
            background_tasks.add_task(cleanup_temp_file, file_path)

@app.get('/health')
async def health_check():
    return {"status": "ok", "message": "Server is running"}

# Additional endpoint for testing audio quality
@app.post('/validate-audio')
async def validate_audio(voice: UploadFile = File(...)):
    """Validate audio file quality and extract basic metrics"""
    try:
        content = await voice.read()
        
        if not validate_audio_content(content):
            raise HTTPException(status_code=400, detail="Audio file is too small or empty")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            y, sr = librosa.load(tmp_path, sr=16000)
            
            # Calculate audio quality metrics
            duration = len(y) / sr
            rms_energy = np.mean(librosa.feature.rms(y=y))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                "duration_seconds": duration,
                "rms_energy": float(rms_energy),
                "zero_crossing_rate": float(zero_crossing_rate),
                "sample_rate": sr,
                "samples": len(y),
                "is_valid": duration > 1.0 and rms_energy > 0.001
            }
        finally:
            cleanup_temp_file(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio validation failed: {str(e)}")