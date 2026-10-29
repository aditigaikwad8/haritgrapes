"""Shared helpers: hashing, logging, image loading, status rules."""
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import config

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ------------------------------------------------------------------ general
def today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_logger(name: str, logfile: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seeds(seed: int = config.SEED, use_tf: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_tf:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ hashing
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dhash(im: Image.Image):
    """64-bit difference hash. Returns (8 uint8 bytes, is_flat)."""
    small = np.asarray(im.convert("L").resize((9, 8), _RESAMPLE), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    packed = np.packbits(bits)                       # 8 bytes
    flat = float(small.std()) < config.FLAT_IMAGE_STD
    return packed, flat


# ------------------------------------------------------------------- images
def list_images(folder: Path) -> list:
    """Sorted image files below folder. Missing folder -> empty list."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in config.RAW_IMAGE_EXTS]
    return sorted(files, key=lambda p: p.as_posix())


def shrink(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    longest = max(w, h)
    if longest > max_side:
        s = max_side / longest
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), _RESAMPLE)
    return im


def load_rgb(path: Path, max_side: int = None) -> Image.Image:
    """Open, apply EXIF rotation, convert to RGB (drops EXIF/GPS), optionally shrink."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    if max_side:
        im = shrink(im, max_side)
    return im


def load_for_model(path: Path) -> np.ndarray:
    """Raw 0-255 float32 array (224,224,3). The model does its own scaling.

    Same route as the training data: image stored at <=512 px, then bilinear
    resize (what keras image_dataset_from_directory does). Used by evaluate,
    the sanity checks and later predict.py so every consumer sees identical pixels.
    """
    import tensorflow as tf
    im = load_rgb(path, max_side=config.STORE_MAX_SIDE)
    arr = np.asarray(im, dtype=np.float32)
    return tf.image.resize(arr, config.IMG_SIZE, method="bilinear").numpy()


def predict_paths(model, paths, batch_size: int = 32, logger=None):
    """Return (probs [N_ok, C], kept_indices). Unreadable files are skipped."""
    probs, kept, batch, idx = [], [], [], []

    def flush():
        if batch:
            probs.append(model.predict(np.stack(batch), verbose=0))
            kept.extend(idx)
            batch.clear()
            idx.clear()

    for i, p in enumerate(paths):
        try:
            arr = load_for_model(p)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("skipping unreadable image %s (%s)", p, exc)
            continue
        batch.append(arr)
        idx.append(i)
        if len(batch) == batch_size:
            flush()
    flush()
    if probs:
        return np.concatenate(probs, axis=0), kept
    return np.zeros((0, config.NUM_CLASSES), dtype=np.float32), kept


# ------------------------------------------------------------ status rules
def decide_status(probs, class_names=None) -> dict:
    """Doc 04 decision rules. probs: 1-D probability vector in model class order.

    Returns label, confidence, margin, status, top3. predict.py reuses this.
    """
    class_names = class_names or config.CLASS_NAMES
    t = config.THRESHOLDS
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs)
    top1, top2 = int(order[0]), int(order[1])
    conf = float(probs[top1])
    margin = conf - float(probs[top2])
    label = class_names[top1]

    if label == config.NOT_LEAF_CLASS:
        status = "not_a_leaf" if conf >= t["not_a_leaf_min_conf"] else "low_confidence"
    elif conf < t["low_conf_below"]:
        status = "low_confidence"
    else:
        need = t["healthy_min_conf"] if label == config.HEALTHY_CLASS else t["confident_min_conf"]
        if conf >= need and margin >= t["confident_min_margin"]:
            status = "confident"
        else:
            status = "uncertain"
    return {
        "label": label,
        "confidence": conf,
        "margin": margin,
        "status": status,
        "top3": [(class_names[int(i)], float(probs[int(i)])) for i in order[:3]],
    }

def save_joblib_bundle(keras_path, out_path, version_tag):
    """One-file copy of the model: .keras bytes + class names + settings."""
    import joblib
    bundle = {
        "version": version_tag,
        "class_names": config.CLASS_NAMES,
        "img_size": config.IMG_SIZE,
        "thresholds": config.THRESHOLDS,
        "keras_bytes": Path(keras_path).read_bytes(),
    }
    joblib.dump(bundle, out_path, compress=3)


def load_joblib_bundle(path):
    """Return (keras_model, bundle_dict). Only load files you created yourself."""
    import joblib
    import tempfile
    import tensorflow as tf
    bundle = joblib.load(path)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "model.keras"
        tmp.write_bytes(bundle["keras_bytes"])
        model = tf.keras.models.load_model(str(tmp), compile=False)
    return model, bundle

