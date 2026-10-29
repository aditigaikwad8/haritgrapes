"""Train MobileNetV2 on data/split and save a versioned model.

One script, one run (docs/02_TRAINING.md):
    python -m src.train                 # saves models/versions/v1_<date>.keras
    python -m src.train --version v2

Design guarantees
  * preprocessing (scale to [-1,1]) is INSIDE the model; inputs are raw 0-255 pixels.
  * augmentation is inside the model and only active while training.
  * val/test datasets are never shuffled.
  * class weights come from TRAIN counts; seeds are logged.
  * sanity checks run, and a model that fails them is NOT saved as a version.
The test set is never touched here (see src/evaluate.py).
"""
import argparse
import json
import platform
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src import config, utils  # noqa: E402

log = utils.get_logger("train", config.LOGS_DIR / "train.log")


# --------------------------------------------------------------------- data
def train_counts() -> list:
    counts = []
    for cls in config.CLASS_NAMES:
        n = len(utils.list_images(config.SPLIT_DIR / "train" / cls))
        if n == 0:
            sys.exit(f"ERROR: no training images for {cls}. Run: python -m src.data_prep build")
        counts.append(n)
    return counts


def make_dataset(split: str, batch_size: int, shuffle: bool):
    ds = keras.utils.image_dataset_from_directory(
        str(config.SPLIT_DIR / split),
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,       # fixed order, never inferred
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMG_SIZE,
        shuffle=shuffle,                      # False for val/test
        seed=config.SEED if shuffle else None,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def labels_of(ds) -> np.ndarray:
    return np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])


# -------------------------------------------------------------------- model
def build_model(pretrained: bool = True):
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1, fill_mode="reflect"),
        layers.RandomZoom(0.15, fill_mode="reflect"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")

    inputs = keras.Input(shape=(*config.IMG_SIZE, config.IMG_CHANNELS), name="image")
    x = augmentation(inputs)                                   # training only
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)   # MobileNetV2 scaling
    base = keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, config.IMG_CHANNELS),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(config.DROPOUT, name="dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="haritgrapes_mobilenetv2"), base


def compile_model(model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )


# ------------------------------------------------------------------ helpers
def merge_history(parts) -> dict:
    out = {}
    for h in parts:
        for k, v in h.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def plot_curves(history: dict, phase1_epochs: int, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, key, title in ((ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")):
        a.plot(history.get(key, []), label="train")
        a.plot(history.get("val_" + key, []), label="val")
        a.axvline(phase1_epochs - 0.5, color="grey", linestyle="--", label="fine-tune starts")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_environment() -> None:
    import PIL
    import sklearn
    lines = [f"python {platform.python_version()}", f"tensorflow {tf.__version__}",
             f"keras {keras.__version__}", f"numpy {np.__version__}",
             f"scikit-learn {sklearn.__version__}", f"pillow {PIL.__version__}",
             f"matplotlib {matplotlib.__version__}",
             f"gpus: {[g.name for g in tf.config.list_physical_devices('GPU')] or 'none'}"]
    (config.REPORTS_DIR / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sanity_train_images(model) -> float:
    """Classify 20 train images through the same loader evaluate/predict use."""
    import random
    rng = random.Random(config.SEED)
    paths, y = [], []
    for i, cls in enumerate(config.CLASS_NAMES):
        files = utils.list_images(config.SPLIT_DIR / "train" / cls)
        pick = rng.sample(files, min(config.SANITY_TRAIN_IMAGES_PER_CLASS, len(files)))
        paths += pick
        y += [i] * len(pick)
    probs, kept = utils.predict_paths(model, paths, logger=log)
    y = np.array(y)[kept]
    return float((np.argmax(probs, axis=1) == y).mean())


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--epochs1", type=int, default=config.PHASE1_EPOCHS)
    ap.add_argument("--epochs2", type=int, default=config.PHASE2_EPOCHS)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="debug only: random weights (offline smoke test)")
    args = ap.parse_args()

    utils.set_seeds(config.SEED)
    log.info("Seeds set: python/numpy/tensorflow = %d", config.SEED)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    write_environment()

    tag = f"{args.version}_{utils.today_stamp()}"
    report_dir = config.REPORTS_DIR / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.VERSIONS_DIR / f"{tag}.keras"
    if model_path.exists():
        sys.exit(f"ERROR: {model_path} already exists. Use --version v2 (never overwrite a version).")

    counts = train_counts()
    total = sum(counts)
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    log.info("Train counts: %s", dict(zip(config.CLASS_NAMES, counts)))
    log.info("Class weights: %s", {config.CLASS_NAMES[i]: round(w, 3) for i, w in class_weight.items()})

    train_ds = make_dataset("train", args.batch_size, shuffle=True)
    val_ds = make_dataset("val", args.batch_size, shuffle=False)
    y_val = labels_of(val_ds)
    log.info("Val label counts: %s", dict(zip(config.CLASS_NAMES, np.bincount(y_val, minlength=config.NUM_CLASSES).tolist())))

    model, base = build_model(pretrained=not args.no_pretrained)
    compile_model(model, config.PHASE1_LR)

    # sanity 1: untrained head should be near chance
    _, acc0 = model.evaluate(val_ds, verbose=0)
    log.info("Sanity 1: untrained accuracy on val = %.3f (chance = %.2f)", acc0, 1 / config.NUM_CLASSES)
    if acc0 > 0.5:
        log.warning("Untrained accuracy is suspiciously high. Check for label leakage.")

    # phase 1: head only
    log.info("PHASE 1: training the new head (base frozen)")
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs1, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PHASE1_PATIENCE,
                                                 restore_best_weights=True)],
        verbose=2)

    # phase 2: fine-tune top layers, BatchNorm stays frozen
    log.info("PHASE 2: fine-tuning top %d layers", config.FINE_TUNE_LAYERS)
    base.trainable = True
    for layer in base.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, config.PHASE2_LR)
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs2, class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PHASE2_PATIENCE,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ],
        verbose=2)

    history = merge_history([h1.history, h2.history])
    n_phase1 = len(h1.history["loss"])
    plot_curves(history, n_phase1, report_dir / "training_curves.png")
    utils.write_json(report_dir / "history.json", history)

    # ---- sanity checks on the final model
    probs = model.predict(val_ds, verbose=0)
    pred = np.argmax(probs, axis=1)
    val_acc = float(accuracy_score(y_val, pred))
    val_f1 = float(f1_score(y_val, pred, average="macro", labels=list(range(config.NUM_CLASSES)), zero_division=0))
    dist = np.bincount(pred, minlength=config.NUM_CLASSES)
    share = dist.max() / max(1, dist.sum())
    log.info("Val accuracy %.4f | macro-F1 %.4f", val_acc, val_f1)
    log.info("Sanity 3: predicted class distribution on val: %s", dict(zip(config.CLASS_NAMES, dist.tolist())))
    train_acc20 = sanity_train_images(model)
    log.info("Sanity 2: accuracy on 20 train images through the predict path = %.2f", train_acc20)

    problems = []
    if train_acc20 < config.SANITY_MIN_TRAIN_ACC:
        problems.append(f"only {train_acc20:.0%} of 20 train images classified correctly (bug in preprocessing/class order?)")
    if (dist == 0).any():
        problems.append("some classes are never predicted: " +
                        str([config.CLASS_NAMES[i] for i in np.where(dist == 0)[0]]))
    if share > config.SANITY_MAX_SINGLE_CLASS_SHARE:
        problems.append(f"one class takes {share:.0%} of all predictions")
    if problems:
        bad = config.VERSIONS_DIR / f"FAILED_SANITY_{tag}.keras"
        model.save(str(bad))
        log.error("SANITY CHECKS FAILED -> not saved as a version (kept for debugging as %s):", bad.name)
        for p in problems:
            log.error("  - %s", p)
        sys.exit(1)

    # ---- save
    model.save(str(model_path))
    joblib_path = config.VERSIONS_DIR / f"{tag}.joblib"
    utils.save_joblib_bundle(model_path, joblib_path, tag)
    log.info("Also saved %s", joblib_path.name)
    if config.CLASS_NAMES_FILE.exists() and utils.read_json(config.CLASS_NAMES_FILE) != config.CLASS_NAMES:
        sys.exit("ERROR: models/versions/class_names.json differs from config.CLASS_NAMES")
    utils.write_json(config.CLASS_NAMES_FILE, config.CLASS_NAMES)
    utils.write_json(report_dir / "class_names.json", config.CLASS_NAMES)
    snap = config.snapshot()
    snap.update({
        "version_tag": tag, "model_file": model_path.name, "seed": config.SEED,
        "train_counts": dict(zip(config.CLASS_NAMES, counts)),
        "class_weights": {config.CLASS_NAMES[i]: w for i, w in class_weight.items()},
        "phase1_epochs_run": n_phase1, "phase2_epochs_run": len(h2.history["loss"]),
        "val_accuracy": val_acc, "val_macro_f1": val_f1,
        "val_pred_distribution": dict(zip(config.CLASS_NAMES, dist.tolist())),
        "sanity_untrained_val_acc": float(acc0), "sanity_train20_acc": train_acc20,
        "pretrained": not args.no_pretrained,
    })
    utils.write_json(report_dir / "config_snapshot.json", snap)
    log.info("Saved %s, models/versions/class_names.json and reports/%s/", model_path.name, tag)
    log.info("Next: python -m src.evaluate --model %s", model_path)


if __name__ == "__main__":
    main()