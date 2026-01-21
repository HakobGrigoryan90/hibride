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
import pandas as pd
from collections import deque
from datetime import datetime
import math


app = FastAPI(title="Multi-appliance NILM API", version="11.3")

# Optional: helps Swagger UI / browser clients call the API cross-origin.
# See FastAPI CORSMiddleware docs for correct configuration. [web:24]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multi_appliance_nilm.h5")
SCALERS_PATH = os.path.join(MODEL_DIR, "scalers_multi_appliance.pkl")
SEQUENCE_LENGTH = 30

FRIDGE_NAME = "Fridge-Freezer"
ON_PROB_THRESHOLD = 0.5
RECON_WEIGHT = 5.0  # must match training


# ========= Pydantic models =========
class TimePoint(BaseModel):
    time: str
    aggregate: float


class InferenceRequest(BaseModel):
    # Streaming mode: exactly 30 points to initialize; then single_point repeatedly.
    full_sequence: Optional[List[TimePoint]] = None
    single_point: Optional[TimePoint] = None


class SequenceRequest(BaseModel):
    # Batch mode: provide whole series; API returns predictions per row (after warmup).
    points: List[TimePoint]


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


# ========= Feature definition =========
FEATURE_COLS = [
    "Aggregate",
    "Agg_diff1",
    "Agg_diff2",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]


def compute_features(tp: TimePoint, prev_tp: Optional[TimePoint]) -> np.ndarray:
    dt = datetime.fromisoformat(tp.time)
    hour = dt.hour
    dow = dt.weekday()

    sin_hour = math.sin(2 * math.pi * hour / 24.0)
    cos_hour = math.cos(2 * math.pi * hour / 24.0)
    sin_dow = math.sin(2 * math.pi * dow / 7.0)
    cos_dow = math.cos(2 * math.pi * dow / 7.0)

    if prev_tp is None:
        diff1 = 0.0
        diff2 = 0.0
    else:
        diff1 = tp.aggregate - prev_tp.aggregate
        diff2 = diff1

    return np.array(
        [
            tp.aggregate,
            diff1,
            diff2,
            sin_hour,
            cos_hour,
            sin_dow,
            cos_dow,
        ],
        dtype=float,
    )


# ========= CSV parsing =========
def csv_to_timepoints(csv_content: bytes) -> List[TimePoint]:
    """
    Required column: time
    Aggregate accepted aliases (case-insensitive):
      aggregate, aggregate_power, mains, mains_power, total, total_power
    """
    try:
        text = csv_content.decode("utf-8")
    except UnicodeDecodeError:
        text = csv_content.decode("utf-8-sig")

    df = pd.read_csv(io.StringIO(text))
    if df.shape[0] == 0:
        return []

    cols_lower = {c.lower(): c for c in df.columns}

    if "time" not in cols_lower:
        raise ValueError("CSV must contain a 'time' column")
    time_col = cols_lower["time"]

    agg_candidates = [
        "aggregate",
        "aggregate_power",
        "mains",
        "mains_power",
        "total",
        "total_power",
    ]
    agg_col = None
    for cand in agg_candidates:
        if cand in cols_lower:
            agg_col = cols_lower[cand]
            break
    if agg_col is None:
        raise ValueError("CSV must contain an aggregate column (e.g., 'aggregate')")

    df = df[[time_col, agg_col]].dropna()
    df = df.rename(columns={time_col: "time", agg_col: "aggregate"})

    tps: List[TimePoint] = []
    for _, row in df.iterrows():
        tps.append(TimePoint(time=str(row["time"]), aggregate=float(row["aggregate"])))
    return tps


# ========= Load model & scalers =========
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALERS_PATH):
    raise RuntimeError("Model or scalers missing.")


@register_keras_serializable(package="nilm", name="sum_power_fn")
def sum_power_fn(t):
    return tf.reduce_sum(t, axis=-1, keepdims=True)


keras.config.enable_unsafe_deserialization()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
)

scalers_info = joblib.load(SCALERS_PATH)
feature_scaler = scalers_info["feature_scaler"]
target_scaler = scalers_info["target_scaler"]
appliances = scalers_info["appliances"]
on_thresholds_scaled = scalers_info["on_thresholds_scaled"]


# === Reattach custom losses and total_loss_fn (for finetuning) ===
reg_loss = tf.keras.losses.Huber()
cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
recon_loss = tf.keras.losses.MeanAbsoluteError()


def total_loss(y_true, y_pred):
    y_power_true, y_onoff_true, y_agg_true = y_true
    y_power_pred, y_onoff_pred, y_sum_pred = y_pred

    l_reg = reg_loss(y_power_true, y_power_pred)
    l_cls = cls_loss(y_onoff_true, y_onoff_pred)
    l_recon = recon_loss(y_agg_true, y_sum_pred)

    return l_reg + l_cls + RECON_WEIGHT * l_recon


model.reg_loss_fn = reg_loss
model.cls_loss_fn = cls_loss
model.recon_loss_fn = recon_loss
model.total_loss_fn = total_loss


# ========= Global state for true streaming (/predict/) =========
stream_buffer = deque(maxlen=SEQUENCE_LENGTH)
stream_prev_tp: Optional[TimePoint] = None


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "NILM multi-appliance API (streaming + full-sequence batch)",
        "sequence_length": SEQUENCE_LENGTH,
        "appliances": appliances,
        "endpoints": [
            "POST /predict/ (streaming: init 30, then single_point)",
            "POST /predict/sequence/ (JSON: whole series -> per-row outputs)",
            "POST /predict/csv/sequence/ (CSV: whole series -> per-row outputs)",
            "POST /finetune/",
        ],
    }


# ========= Prediction helpers =========
def predict_from_window(window: deque, agg_val: float) -> List[AppliancePrediction]:
    if len(window) < SEQUENCE_LENGTH:
        raise HTTPException(400, "Buffer not full")

    X = np.array(window).reshape(1, SEQUENCE_LENGTH, -1)
    power_scaled, onoff_prob, _ = model.predict(X, verbose=0)

    power_scaled = power_scaled[0]
    onoff_prob = onoff_prob[0]

    full_scaled_vector = np.zeros((1, len(appliances) + 1))
    full_scaled_vector[0, : len(appliances)] = power_scaled
    full_inverse = target_scaler.inverse_transform(full_scaled_vector)[0]

    results: List[AppliancePrediction] = []
    for idx, appliance in enumerate(appliances):
        pred_power = max(0.0, float(full_inverse[idx]))
        prob = float(onoff_prob[idx])
        if appliance != FRIDGE_NAME and prob < ON_PROB_THRESHOLD:
            pred_power = 0.0
        results.append(AppliancePrediction(appliance=appliance, prediction=pred_power))

    # Optional: rescale so sum matches aggregate
    total = sum(r.prediction for r in results)
    if total > 0:
        scale = agg_val / total
        for r in results:
            r.prediction *= scale

    return results


def predict_sequence_points(points: List[TimePoint]) -> List[PredictionFrame]:
    """
    Full-sequence mode that mimics:
      1) init with first 30 points
      2) then stream point-by-point and predict each step
    Returns 1 PredictionFrame per point starting at index SEQUENCE_LENGTH-1.
    """
    if len(points) < SEQUENCE_LENGTH:
        raise HTTPException(400, f"Need at least {SEQUENCE_LENGTH} points")

    local_window = deque(maxlen=SEQUENCE_LENGTH)
    prev_tp: Optional[TimePoint] = None

    frames: List[PredictionFrame] = []
    for tp in points:
        feats = compute_features(tp, prev_tp).reshape(1, -1)
        feats_scaled = feature_scaler.transform(feats)[0]
        local_window.append(feats_scaled)
        prev_tp = tp

        if len(local_window) < SEQUENCE_LENGTH:
            continue

        preds = predict_from_window(local_window, tp.aggregate)
        frames.append(PredictionFrame(time=tp.time, predictions=preds))

    return frames


# ========= Streaming predict =========
@app.post("/predict/", response_model=List[AppliancePrediction])
def predict(body: InferenceRequest):
    global stream_prev_tp

    if body.full_sequence is None and body.single_point is None:
        raise HTTPException(400, "Provide full_sequence or single_point")

    # init with full_sequence (must be exactly 30 for streaming state)
    if body.full_sequence is not None:
        if len(body.full_sequence) != SEQUENCE_LENGTH:
            raise HTTPException(400, f"full_sequence must have exactly {SEQUENCE_LENGTH} points")

        stream_buffer.clear()
        prev_tp = None
        for tp in body.full_sequence:
            feats = compute_features(tp, prev_tp).reshape(1, -1)
            feats_scaled = feature_scaler.transform(feats)[0]
            stream_buffer.append(feats_scaled)
            prev_tp = tp
        stream_prev_tp = prev_tp

    # append single point
    if body.single_point is not None:
        if stream_prev_tp is None:
            raise HTTPException(400, "Initialize using full_sequence first")

        tp = body.single_point
        feats = compute_features(tp, stream_prev_tp).reshape(1, -1)
        feats_scaled = feature_scaler.transform(feats)[0]
        stream_buffer.append(feats_scaled)
        stream_prev_tp = tp

    # predict for current last point
    if body.single_point is not None:
        agg_val = body.single_point.aggregate
    else:
        agg_val = body.full_sequence[-1].aggregate

    return predict_from_window(stream_buffer, agg_val)


# ========= Full-sequence JSON prediction =========
@app.post("/predict/sequence/", response_model=List[PredictionFrame])
def predict_sequence(req: SequenceRequest):
    return predict_sequence_points(req.points)


# ========= Full-sequence CSV prediction =========
@app.post("/predict/csv/sequence/", response_model=List[PredictionFrame])
async def predict_csv_sequence(csv_file: UploadFile = File(...)):
    # UploadFile reading is async per FastAPI docs. [web:41][web:42]
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be a .csv")

    content = await csv_file.read()

    try:
        points = csv_to_timepoints(content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return predict_sequence_points(points)


# ========= Finetune =========
@app.post("/finetune/")
def finetune(req: FineTuneRequest):
    num_appliances = len(appliances)

    if len(req.points) < SEQUENCE_LENGTH:
        raise HTTPException(400, "Not enough points to build sequences for finetuning")

    aggs = np.array([p.aggregate for p in req.points], dtype="float32")
    targets = np.array([p.appliance_powers for p in req.points], dtype="float32")

    feats_raw = []
    prev_tp: Optional[TimePoint] = None
    for p in req.points:
        tp_simple = TimePoint(time=p.time, aggregate=p.aggregate)
        feats_raw.append(compute_features(tp_simple, prev_tp))
        prev_tp = tp_simple
    feats_raw = np.array(feats_raw, dtype="float32")

    feats_scaled = feature_scaler.transform(feats_raw)

    full_targets = np.concatenate([targets, aggs.reshape(-1, 1)], axis=1)
    full_targets_scaled = target_scaler.transform(full_targets)
    y_power_scaled = full_targets_scaled[:, :num_appliances]
    y_agg_scaled = full_targets_scaled[:, num_appliances : num_appliances + 1]

    X_list, Y_power_list, Y_agg_list = [], [], []
    for i in range(len(req.points) - SEQUENCE_LENGTH + 1):
        X_list.append(feats_scaled[i : i + SEQUENCE_LENGTH])
        Y_power_list.append(y_power_scaled[i + SEQUENCE_LENGTH - 1])
        Y_agg_list.append(y_agg_scaled[i + SEQUENCE_LENGTH - 1])

    X_all = np.array(X_list, dtype="float32")
    Y_power_all = np.array(Y_power_list, dtype="float32")
    Y_agg_all = np.array(Y_agg_list, dtype="float32")

    # on/off labels
    Y_onoff = np.zeros_like(Y_power_all, dtype="float32")
    for j, appliance in enumerate(appliances):
        if appliance == FRIDGE_NAME:
            Y_onoff[:, j] = 1.0
        else:
            thr = on_thresholds_scaled[appliance]
            Y_onoff[:, j] = (Y_power_all[:, j] > thr).astype("float32")

    if X_all.shape[0] == 0:
        raise HTTPException(400, "Not enough points to build any training window")

    ds = (
        tf.data.Dataset.from_tensor_slices((X_all, Y_power_all, Y_onoff, Y_agg_all))
        .shuffle(len(X_all))
        .batch(req.batch_size)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def finetune_step(x, y_p, y_c, y_a):
        with tf.GradientTape() as tape:
            p_pred, c_pred, s_pred = model(x, training=True)
            loss = model.total_loss_fn([y_p, y_c, y_a], [p_pred, c_pred, s_pred])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(req.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_b, y_p_b, y_c_b, y_a_b in ds:
            loss_b = finetune_step(x_b, y_p_b, y_c_b, y_a_b)
            epoch_loss += float(loss_b)
            n_batches += 1
        print(f"[finetune] epoch {epoch+1}/{req.epochs}, loss={epoch_loss / max(n_batches,1):.6f}")

    model.save(MODEL_PATH)

    return {
        "status": "ok",
        "epochs": req.epochs,
        "num_sequences": int(X_all.shape[0]),
    }
