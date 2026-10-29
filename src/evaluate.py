"""Evaluate a trained model on test + field_test (+ out-of-scope check).

    python -m src.evaluate                       # newest models/versions/v*.keras
    python -m src.evaluate --model models/versions/v1_20261029.keras

Writes reports/<version>/ : report.md, metrics_*.json, predictions_*.csv,
confusion_matrix_*.csv/.png, worst_mistakes_*.png, confidence_hist_*.png,
out_of_scope.json.  Never use these numbers to tune on the test set:
thresholds are chosen on validation + field_test only.
"""
import argparse
import csv
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)

from src import config, utils  # noqa: E402

log = utils.get_logger("evaluate", config.LOGS_DIR / "evaluate.log")
NAMES = config.CLASS_NAMES
NL = NAMES.index(config.NOT_LEAF_CLASS)


# --------------------------------------------------------------- collection
def collect_labeled(root: Path):
    paths, y = [], []
    for i, cls in enumerate(NAMES):
        for p in utils.list_images(Path(root) / cls):
            paths.append(p)
            y.append(i)
    return paths, np.array(y, dtype=int)


def find_latest_model() -> Path:
    cands = sorted(p for p in config.VERSIONS_DIR.glob("v*.keras") if not p.name.startswith("FAILED"))
    if not cands:
        sys.exit("ERROR: no model found in models/versions. Run: python -m src.train")
    return cands[-1]


# ----------------------------------------------------------------- plotting
def plot_confusion(cm, path, title) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, m, t, fmt in ((ax[0], cm, f"{title} (counts)", "d"), (ax[1], norm, f"{title} (row-normalised)", ".2f")):
        a.imshow(m, cmap="Blues", vmin=0)
        a.set_xticks(range(len(NAMES)))
        a.set_yticks(range(len(NAMES)))
        a.set_xticklabels(NAMES, rotation=45, ha="right")
        a.set_yticklabels(NAMES)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(t)
        thr = m.max() / 2 if m.max() > 0 else 0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                a.text(j, i, format(m[i, j], fmt), ha="center", va="center",
                       color="white" if m[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_worst(paths, y, pred, conf, path, title, k: int = 20) -> int:
    wrong = np.where(pred != y)[0]
    if len(wrong) == 0:
        return 0
    worst = wrong[np.argsort(-conf[wrong])][:k]
    cols = 5
    rows = (len(worst) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, worst):
        try:
            a.imshow(utils.load_rgb(paths[i], max_side=300))
        except Exception:  # noqa: BLE001
            pass
        a.set_title(f"T: {NAMES[y[i]]}\nP: {NAMES[pred[i]]} {conf[i]:.2f}", fontsize=8)
    fig.suptitle(f"{title}: {len(worst)} most confident mistakes")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return len(worst)


def plot_conf_hist(conf, correct, path, title) -> None:
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[correct], bins=bins, alpha=0.7, label=f"correct ({int(correct.sum())})")
    if (~correct).any():
        ax.hist(conf[~correct], bins=bins, alpha=0.7, label=f"wrong ({int((~correct).sum())})")
    ax.set_xlabel("confidence of predicted class")
    ax.set_ylabel("images")
    ax.set_title(f"{title}: confidence, correct vs wrong")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ metrics
def evaluate_set(name, paths, y, probs, out_dir: Path) -> dict:
    pred = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(pred)), pred]
    labels = list(range(len(NAMES)))
    present = sorted(set(y.tolist()))
    rep = classification_report(y, pred, labels=labels, target_names=NAMES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    m = {
        "n_images": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(rep[c]["support"])} for c in NAMES},
        "confusion_matrix": cm.tolist(),
        "pred_distribution": {c: int((pred == i).sum()) for i, c in enumerate(NAMES)},
    }
    is_nl = y == NL
    m["not_leaf"] = {
        "recall": float((pred[is_nl] == NL).mean()) if is_nl.any() else None,
        "false_accept_rate": float((pred[is_nl] != NL).mean()) if is_nl.any() else None,   # non-leaf accepted as a grape class
        "false_reject_rate": float((pred[~is_nl] == NL).mean()) if (~is_nl).any() else None,  # grape leaf rejected as non-leaf
    }
    # files
    np.savetxt(out_dir / f"confusion_matrix_{name}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(NAMES), comments="")
    plot_confusion(cm, out_dir / f"confusion_matrix_{name}.png", name)
    m["n_worst_plotted"] = plot_worst(paths, y, pred, conf, out_dir / f"worst_mistakes_{name}.png", name)
    plot_conf_hist(conf, pred == y, out_dir / f"confidence_hist_{name}.png", name)
    with open(out_dir / f"predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence"] + [f"p_{c}" for c in NAMES])
        for p, t, q, c, pr in zip(paths, y, pred, conf, probs):
            w.writerow([str(p), NAMES[t], NAMES[q], f"{c:.4f}"] + [f"{v:.4f}" for v in pr])
    utils.write_json(out_dir / f"metrics_{name}.json", m)
    return m


def out_of_scope_check(model, out_dir: Path):
    files = utils.list_images(config.OUT_OF_SCOPE_DIR)
    if not files:
        return None
    probs, kept = utils.predict_paths(model, files, logger=log)
    files = [files[i] for i in kept]
    groups = defaultdict(list)
    for p, pr in zip(files, probs):
        sub = p.relative_to(config.OUT_OF_SCOPE_DIR).parts
        groups["ALL"].append(pr)
        if len(sub) > 1:
            groups[sub[0]].append(pr)

    def stats(plist):
        res = [utils.decide_status(p) for p in plist]
        n = len(res)
        return {
            "n": n,
            "predicted_healthy_confident_pct": 100 * sum(r["label"] == config.HEALTHY_CLASS and r["status"] == "confident" for r in res) / n,
            "predicted_healthy_any_pct": 100 * sum(r["label"] == config.HEALTHY_CLASS for r in res) / n,
            "uncertain_or_low_confidence_pct": 100 * sum(r["status"] in ("uncertain", "low_confidence") for r in res) / n,
            "not_a_leaf_pct": 100 * sum(r["status"] == "not_a_leaf" for r in res) / n,
            "forced_into_class_pct": {c: 100 * sum(r["label"] == c for r in res) / n for c in NAMES},
            "status_counts": dict(Counter(r["status"] for r in res)),
        }
    out = {k: stats(v) for k, v in groups.items()}
    utils.write_json(out_dir / "out_of_scope.json", out)
    return out


def measure_latency(model, paths, n: int = 30):
    sample = paths[:n]
    if not sample:
        return None
    for p in sample[:3]:                                  # warm-up
        model.predict(utils.load_for_model(p)[None], verbose=0)
    times = []
    for p in sample:
        t0 = time.perf_counter()
        model.predict(utils.load_for_model(p)[None], verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)), "n": len(times)}


# ------------------------------------------------------------------- report
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def mark(ok):
    return "PASS" if ok else "FAIL"


def class_table(m):
    L = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in NAMES:
        d = m["per_class"][c]
        L.append(f"| {c} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1-score']:.3f} | {d['support']} |")
    return L


def write_report(tag, results, oos, latency, checks, out_dir: Path) -> None:
    L = [f"# Evaluation report: {tag}\n"]
    L.append("## Acceptance targets (v1)\n")
    L.append("| Check | Target | Result | |")
    L.append("|---|---|---|---|")
    for desc, target, value, ok in checks:
        L.append(f"| {desc} | {target} | {value} | {ok} |")
    L.append("")
    for name in ("test", "field_test"):
        m = results.get(name)
        L.append(f"## {name}\n")
        if m is None:
            L.append(f"No images found for {name}.\n")
            continue
        L.append(f"Images: {m['n_images']}. Accuracy: {pct(m['accuracy'])}. Macro-F1: {m['macro_f1']:.3f}\n")
        L += class_table(m)
        nl = m["not_leaf"]
        L.append("")
        L.append(f"Not_a_Grape_Leaf: recall {pct(nl['recall'])}, false-accept rate {pct(nl['false_accept_rate'])} "
                 f"(non-leaf accepted as a grape class), false-reject rate {pct(nl['false_reject_rate'])} "
                 "(grape/other class predicted as non-leaf).\n")
        L.append(f"Predicted class distribution: {m['pred_distribution']}\n")
        L.append(f"Figures: confusion_matrix_{name}.png, worst_mistakes_{name}.png, confidence_hist_{name}.png. "
                 f"Per-image results: predictions_{name}.csv. Look at the worst mistakes for mislabels and look-alikes.\n")
    L.append("## Out-of-scope check (Downy Mildew / Powdery Mildew, never trained on)\n")
    if not oos:
        L.append("No images in data/field_test/out_of_scope/. Collect 20-30 mildew photos and rerun. "
                 "This check is required by the Definition of Done.\n")
    else:
        L.append("Status uses the starting thresholds in src/config.py.\n")
        L.append("| Group | N | Healthy + confident (dangerous) | Healthy (any) | Uncertain/low-conf | Not a leaf | Forced into each class |")
        L.append("|---|---|---|---|---|---|---|")
        for g, s in oos.items():
            forced = ", ".join(f"{c} {v:.0f}%" for c, v in s["forced_into_class_pct"].items() if v > 0)
            L.append(f"| {g} | {s['n']} | {s['predicted_healthy_confident_pct']:.0f}% | "
                     f"{s['predicted_healthy_any_pct']:.0f}% | {s['uncertain_or_low_confidence_pct']:.0f}% | "
                     f"{s['not_a_leaf_pct']:.0f}% | {forced} |")
        L.append("")
    L.append("## Latency (load + preprocess + predict, one image at a time)\n")
    if latency:
        L.append(f"Median {latency['median_ms']:.0f} ms, p95 {latency['p95_ms']:.0f} ms over {latency['n']} images "
                 f"(target <= {config.LATENCY_TARGET_MS} ms desktop).\n")
    L.append("## Notes\n")
    L.append("- GVLiD is a single source (one phone, one date): field_test accuracy is the honest measure.")
    L.append("- Choose thresholds and calibration on validation + field_test, never on the test set.")
    L.append("- If targets are missed: inspect confusions, clean labels, add varied data, retrain. Do not tune on test.")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    import tensorflow as tf
    model_path = args.model or find_latest_model()
    if not model_path.exists():
        sys.exit(f"ERROR: {model_path} not found")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # the class order must come from the saved file, and must match the model
    if not config.CLASS_NAMES_FILE.exists():
        sys.exit("ERROR: models/versions/class_names.json missing (written by train.py)")
    saved_names = utils.read_json(config.CLASS_NAMES_FILE)
    if saved_names != NAMES:
        sys.exit(f"ERROR: class_names.json {saved_names} != config.CLASS_NAMES {NAMES}")
    if model.output_shape[-1] != len(saved_names):
        sys.exit(f"ERROR: model has {model.output_shape[-1]} outputs but {len(saved_names)} class names")

    tag = model_path.stem
    out_dir = config.REPORTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Evaluating %s", model_path.name)

    results, test_paths = {}, []
    for name, root in (("test", config.SPLIT_DIR / "test"), ("field_test", config.FIELD_TEST_DIR)):
        paths, y = collect_labeled(root)
        if name == "test":
            test_paths = paths
        if not paths:
            log.warning("No images for %s (%s)", name, root)
            continue
        probs, kept = utils.predict_paths(model, paths, batch_size=args.batch_size, logger=log)
        paths = [paths[i] for i in kept]
        y = y[kept]
        results[name] = evaluate_set(name, paths, y, probs, out_dir)
        log.info("%s: n=%d accuracy=%.4f macro-F1=%.4f", name, len(y),
                 results[name]["accuracy"], results[name]["macro_f1"])

    oos = out_of_scope_check(model, out_dir)
    latency = measure_latency(model, test_paths)

    checks = []
    tm = results.get("test")
    if tm:
        t = config.TARGETS
        checks.append(("Test accuracy", f">= {t['test_accuracy']:.0%}", pct(tm["accuracy"]),
                       mark(tm["accuracy"] >= t["test_accuracy"])))
        for c in NAMES:
            r = tm["per_class"][c]["recall"]
            tgt = t["not_leaf_recall"] if c == config.NOT_LEAF_CLASS else t["class_recall"]
            checks.append((f"Recall {c} (test)", f">= {tgt:.2f}", f"{r:.3f}", mark(r >= tgt)))
    else:
        checks.append(("Test set", "present", "missing", "FAIL"))
    fm = results.get("field_test")
    checks.append(("Field-test accuracy", f"report honestly, aim >= {config.TARGETS['field_accuracy_aim']:.0%}",
                   pct(fm["accuracy"]) if fm else "no field photos yet",
                   ("PASS" if fm["accuracy"] >= config.TARGETS["field_accuracy_aim"] else "BELOW AIM") if fm else "TODO"))
    checks.append(("Out-of-scope mildew check", "reported", "done" if oos else "no images", "PASS" if oos else "TODO"))
    if latency:
        checks.append(("Latency (median)", f"<= {config.LATENCY_TARGET_MS} ms", f"{latency['median_ms']:.0f} ms",
                       mark(latency["median_ms"] <= config.LATENCY_TARGET_MS)))

    write_report(tag, results, oos, latency, checks, out_dir)
    log.info("Report: %s", out_dir / "report.md")
    for desc, target, value, ok in checks:
        log.info("%-32s %-28s %-22s %s", desc, target, value, ok)
    if any(c[3] == "FAIL" for c in checks):
        log.info("Targets missed: inspect confusions and worst mistakes, fix data/labels, retrain. Do NOT tune on test.")


if __name__ == "__main__":
    main()