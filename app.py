from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

import joblib
import os
import io
import csv
from collections import deque
from datetime import datetime
import math

app = FastAPI(title="Multi-appliance NILM API", version="11.4")

# ========= CORS =========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Paths & constants =========
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multi_appliance_nilm.h5")
SCALERS_PATH = os.path.join(MODEL_DIR, "scalers_multi_appliance.pkl")
SEQUENCE_LENGTH = 30

FRIDGE_NAME = "Fridge-Freezer"
ON_PROB_THRESHOLD = 0.5
RECON_WEIGHT = 5.0

# ========= Pydantic models =========
class TimePoint(BaseModel):
    time: str
    aggregate: float

class InferenceRequest(BaseModel):
    single_point: TimePoint

class AppliancePrediction(BaseModel):
    appliance: str
    prediction: float

class PredictionFrame(BaseModel):
    time: str
    predictions: List[AppliancePrediction]

class TimePointWithTargets(BaseModel):
    time: str
    aggregate: float
    appliance_powers: List[float]

class FineTuneRequest(BaseModel):
    points: List[TimePointWithTargets]
    epochs: int = 3
    batch_size: int = 16

# ========= Feature engineering =========
def compute_features(tp: TimePoint, prev_tp: Optional[TimePoint]) -> np.ndarray:
    dt = datetime.fromisoformat(tp.time)
    hour = dt.hour
    dow = dt.weekday()

    sin_hour = math.sin(2 * math.pi * hour / 24.0)
    cos_hour = math.cos(2 * math.pi * hour / 24.0)
    sin_dow = math.sin(2 * math.pi * dow / 7.0)
    cos_dow = math.cos(2 * math.pi * dow / 7.0)

    diff1 = tp.aggregate - prev_tp.aggregate if prev_tp else 0.0
    diff2 = diff1

    return np.array([
        tp.aggregate, diff1, diff2,
        sin_hour, cos_hour, sin_dow, cos_dow
    ], dtype="float32")

# ========= CSV parsing =========
def csv_to_timepoints(csv_content: bytes) -> List[TimePoint]:
    try:
        text = csv_content.decode("utf-8")
    except UnicodeDecodeError:
        text = csv_content.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []

    cols_lower = {k.lower(): k for k in rows[0].keys()}
    if "time" not in cols_lower:
        raise ValueError("CSV must contain a 'time' column")
    time_col = cols_lower["time"]

    agg_candidates = ["aggregate", "aggregate_power", "mains", "mains_power", "total", "total_power"]
    agg_col = next((cols_lower[c] for c in agg_candidates if c in cols_lower), None)
    if not agg_col:
        raise ValueError("CSV must contain an aggregate column")

    tps = []
    for row in rows:
        try:
            tps.append(TimePoint(time=str(row[time_col]), aggregate=float(row[agg_col])))
        except:
            continue
    return tps

# ========= Load model & scalers =========
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALERS_PATH):
    raise RuntimeError("Model or scalers missing")

@register_keras_serializable(package="nilm", name="sum_power_fn")
def sum_power_fn(t): return tf.reduce_sum(t, axis=-1, keepdims=True)

keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

scalers_info = joblib.load(SCALERS_PATH)
feature_scaler = scalers_info["feature_scaler"]
target_scaler = scalers_info["target_scaler"]
appliances = scalers_info["appliances"]
on_thresholds_scaled = scalers_info["on_thresholds_scaled"]

# ========= Losses =========
reg_loss = tf.keras.losses.Huber()
cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
recon_loss = tf.keras.losses.MeanAbsoluteError()
def total_loss(y_true, y_pred):
    y_power_true, y_onoff_true, y_agg_true = y_true
    y_power_pred, y_onoff_pred, y_sum_pred = y_pred
    return reg_loss(y_power_true, y_power_pred) + cls_loss(y_onoff_true, y_onoff_pred) + RECON_WEIGHT*recon_loss(y_agg_true, y_sum_pred)
model.total_loss_fn = total_loss

# ========= Streaming buffer =========
stream_buffer = deque(maxlen=SEQUENCE_LENGTH)
stream_prev_tp: Optional[TimePoint] = None

@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "NILM multi-appliance API", "sequence_length": SEQUENCE_LENGTH, "appliances": appliances}

# ========= Prediction helpers =========
def predict_from_window(window: deque, agg_val: float) -> List[AppliancePrediction]:
    X = np.array(window).reshape(1, SEQUENCE_LENGTH, -1)
    power_scaled, onoff_prob, _ = model.predict(X, verbose=0)
    power_scaled, onoff_prob = power_scaled[0], onoff_prob[0]

    full_scaled = np.zeros((1, len(appliances)+1))
    full_scaled[0,:len(appliances)] = power_scaled
    full_inverse = target_scaler.inverse_transform(full_scaled)[0]

    results = []
    for i, appliance in enumerate(appliances):
        pred = max(0.0, float(full_inverse[i]))
        if appliance != FRIDGE_NAME and onoff_prob[i] < ON_PROB_THRESHOLD:
            pred = 0.0
        results.append(AppliancePrediction(appliance=appliance, prediction=pred))

    total = sum(r.prediction for r in results)
    if total>0:
        scale = agg_val/total
        for r in results: r.prediction *= scale
    return results

def predict_sequence_points(points: List[TimePoint]) -> List[PredictionFrame]:
    if len(points) < SEQUENCE_LENGTH:
        raise HTTPException(400, "Not enough points")
    window, prev_tp, frames = deque(maxlen=SEQUENCE_LENGTH), None, []
    for tp in points:
        feats_scaled = feature_scaler.transform(compute_features(tp, prev_tp).reshape(1,-1))[0]
        window.append(feats_scaled)
        prev_tp = tp
        if len(window)==SEQUENCE_LENGTH:
            frames.append(PredictionFrame(time=tp.time, predictions=predict_from_window(window,tp.aggregate)))
    return frames

# ========= /predict/csv endpoint =========
@app.post("/predict/csv/")
async def predict_csv(csv_file: UploadFile = File(...)):
    global stream_buffer, stream_prev_tp
    content = await csv_file.read()
    points = csv_to_timepoints(content)
    if len(points)<SEQUENCE_LENGTH:
        raise HTTPException(400,f"CSV must have at least {SEQUENCE_LENGTH} points")
    # Fill buffer automatically
    stream_buffer.clear()
    prev_tp = None
    for tp in points[-SEQUENCE_LENGTH:]:
        feats_scaled = feature_scaler.transform(compute_features(tp, prev_tp).reshape(1,-1))[0]
        stream_buffer.append(feats_scaled)
        prev_tp = tp
    stream_prev_tp = prev_tp
    return predict_sequence_points(points)

# ========= /predict single point endpoint =========
@app.post("/predict/")
def predict(req: InferenceRequest):
    global stream_buffer, stream_prev_tp
    tp = req.single_point
    if not stream_buffer or stream_prev_tp is None:
        raise HTTPException(400,"Buffer not initialized. Upload CSV first.")
    feats_scaled = feature_scaler.transform(compute_features(tp, stream_prev_tp).reshape(1,-1))[0]
    stream_buffer.append(feats_scaled)
    stream_prev_tp = tp
    return predict_from_window(stream_buffer, tp.aggregate)

# ========= Finetune endpoint =========
@app.post("/finetune/")
def finetune(req: FineTuneRequest):
    num_appliances = len(appliances)
    aggs = np.array([p.aggregate for p in req.points],dtype="float32")
    targets = np.array([p.appliance_powers for p in req.points],dtype="float32")
    feats, prev = [], None
    for p in req.points:
        feats.append(compute_features(TimePoint(time=p.time, aggregate=p.aggregate), prev))
        prev = TimePoint(time=p.time, aggregate=p.aggregate)
    feats = feature_scaler.transform(np.array(feats))
    full_targets = np.concatenate([targets, aggs[:,None]],axis=1)
    full_targets_scaled = target_scaler.transform(full_targets)

    X,Yp,Ya = [],[],[]
    for i in range(len(feats)-SEQUENCE_LENGTH+1):
        X.append(feats[i:i+SEQUENCE_LENGTH])
        Yp.append(full_targets_scaled[i+SEQUENCE_LENGTH-1,:num_appliances])
        Ya.append(full_targets_scaled[i+SEQUENCE_LENGTH-1,num_appliances:])
    X,Yp,Ya = np.array(X), np.array(Yp), np.array(Ya)

    Yc = np.zeros_like(Yp)
    for j, appliance in enumerate(appliances):
        if appliance==FRIDGE_NAME: Yc[:,j]=1.0
        else: Yc[:,j]=(Yp[:,j]>on_thresholds_scaled[appliance]).astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((X,Yp,Yc,Ya)).batch(req.batch_size)
    opt = tf.keras.optimizers.Adam(1e-4)

    for _ in range(req.epochs):
        for xb, ypb, ycb, yab in ds:
            with tf.GradientTape() as tape:
                pp, cp, sp = model(xb, training=True)
                loss = model.total_loss_fn([ypb, ycb, yab],[pp,cp,sp])
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    model.save(MODEL_PATH)
    return {"status":"ok"}
