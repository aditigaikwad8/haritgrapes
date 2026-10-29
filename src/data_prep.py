"""Check/clean raw images, remove duplicates, split 70/15/15 into data/split.

Commands (run from the project root):
    python -m src.data_prep init      # create field_test/out_of_scope folders
    python -m src.data_prep review    # contact sheets of ~50 random images per class
    python -m src.data_prep build     # clean + de-duplicate + split + reports

Rules followed (docs/01_DATA.md):
  * folder name = label; originals only; GVLiD metadata is NOT used.
  * integrity: unreadable / empty / <100 px / grayscale oddities removed,
    other non-RGB modes converted to RGB.
  * exact duplicates (SHA-256) removed ACROSS classes; copies with conflicting
    labels are removed entirely.
  * near-duplicates (dhash) are grouped; at most NEAR_DUP_KEEP are kept and a
    group is never split over train/val/test.
  * stratified 70/15/15, seed 42, group-aware. Split list saved to CSV.
"""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from src import config, utils

log = utils.get_logger("data_prep", config.LOGS_DIR / "data_prep.log")

GVLID_NAME_PATTERN = ("black rot", "esca", "leaf blight", "healthy")


# ---------------------------------------------------------------- folders
def init_dirs() -> None:
    for cls in config.CLASS_NAMES:
        (config.FIELD_TEST_DIR / cls).mkdir(parents=True, exist_ok=True)
        (config.FEEDBACK_APPROVED_DIR / cls).mkdir(parents=True, exist_ok=True)
    config.OUT_OF_SCOPE_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_OF_SCOPE_DIR / "Downy_Mildew").mkdir(exist_ok=True)
    (config.OUT_OF_SCOPE_DIR / "Powdery_Mildew").mkdir(exist_ok=True)
    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Folders ready. Put real phone photos in data/field_test/<Class>/ "
             "and mildew photos in data/field_test/out_of_scope/<Disease>/.")


def scan_raw() -> dict:
    if not config.RAW_DIR.is_dir():
        sys.exit(f"ERROR: {config.RAW_DIR} not found")
    found = {}
    for cls in config.CLASS_NAMES:
        folder = config.RAW_DIR / cls
        if not folder.is_dir():
            sys.exit(f"ERROR: missing class folder {folder}")
        found[cls] = utils.list_images(folder)
    extra = [p.name for p in config.RAW_DIR.iterdir()
             if p.is_dir() and p.name not in config.CLASS_NAMES]
    if extra:
        log.warning("Ignoring unexpected folders in data/raw: %s", extra)
    skipped = [p for cls in config.CLASS_NAMES
               for p in (config.RAW_DIR / cls).rglob("*")
               if p.is_file() and p.suffix.lower() not in config.RAW_IMAGE_EXTS
               and p.name != ".gitkeep"]
    if skipped:
        log.warning("%d non-image files ignored (e.g. %s)", len(skipped), skipped[0].name)
    return found


def rel(path) -> str:
    return path.relative_to(config.RAW_DIR).as_posix()


# ----------------------------------------------------------------- review
def make_review_sheets(n: int, seed: int, per_sheet: int = 25, cols: int = 5, thumb: int = 200) -> None:
    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    found = scan_raw()
    rng = random.Random(seed)
    for cls in config.CLASS_NAMES:
        files = found[cls]
        sample = rng.sample(files, min(n, len(files)))
        for s in range(0, len(sample), per_sheet):
            chunk = sample[s:s + per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            cap = 18
            sheet = Image.new("RGB", (cols * thumb, rows * (thumb + cap)), "white")
            draw = ImageDraw.Draw(sheet)
            mapping = []
            for k, path in enumerate(chunk):
                x, y = (k % cols) * thumb, (k // cols) * (thumb + cap)
                try:
                    im = utils.load_rgb(path, max_side=thumb)
                    im.thumbnail((thumb, thumb))
                    sheet.paste(im, (x, y + cap))
                except Exception:  # noqa: BLE001
                    draw.rectangle([x, y + cap, x + thumb - 1, y + cap + thumb - 1], outline="red")
                    draw.text((x + 8, y + cap + 8), "UNREADABLE", fill="red")
                draw.text((x + 4, y + 3), f"#{s + k + 1}", fill="black")
                mapping.append(f"{s + k + 1}\t{rel(path)}")
            base = config.REVIEW_DIR / f"{cls}_sheet_{s // per_sheet + 1}"
            sheet.save(base.with_suffix(".jpg"), quality=90)
            base.with_suffix(".txt").write_text("\n".join(mapping) + "\n", encoding="utf-8")
        log.info("Review sheets written for %s (%d images)", cls, len(sample))

    # warn about GVLiD-looking filenames inside Not_a_Grape_Leaf
    suspicious = [p for p in found[config.NOT_LEAF_CLASS]
                  if any(p.name.lower().startswith(k) for k in GVLID_NAME_PATTERN)]
    if suspicious:
        log.warning("%d files in %s look like GVLiD/grape images by name (e.g. %s). "
                    "Check them.", len(suspicious), config.NOT_LEAF_CLASS, suspicious[0].name)

    remove_file = config.REVIEW_DIR / "remove_list.txt"
    if not remove_file.exists():
        remove_file.write_text(
            "# One wrong image per line, as <Class>/<filename> (copy from the sheet .txt files).\n"
            "# Example:  Black_Rot/black rot017.jpg\n"
            "# Lines starting with # are ignored. 'build' deletes nothing from data/raw;\n"
            "# it only leaves these files out of data/split.\n",
            encoding="utf-8")
    log.info("Open the .jpg sheets in reports/review/, list mislabelled images in %s, "
             "then run: python -m src.data_prep build", remove_file)
    log.info("If more than ~2 of 50 are wrong in a class, review that whole class.")


def read_remove_list() -> set:
    f = config.REVIEW_DIR / "remove_list.txt"
    items = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip().replace("\\", "/")
            if s and not s.startswith("#"):
                items.add(s)
    return items


# -------------------------------------------------------------- inspection
def inspect_image(path, cls):
    """Return (record, None) if usable or (None, reason)."""
    try:
        if path.stat().st_size == 0:
            return None, "empty_file"
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            mode, (w, h) = im.mode, im.size
    except Exception:  # noqa: BLE001
        return None, "unreadable"
    if min(w, h) < config.MIN_IMAGE_PX:
        return None, "too_small"
    if mode in config.GRAYSCALE_MODES and cls != config.NOT_LEAF_CLASS:
        return None, "grayscale"
    try:
        pil = utils.load_rgb(path, max_side=config.STORE_MAX_SIDE)
        dh, flat = utils.dhash(pil)
        sha = utils.sha256_file(path)
    except Exception:  # noqa: BLE001
        return None, "unreadable"
    return {"cls": cls, "path": path, "rel": rel(path), "sha": sha, "dh": dh,
            "flat": flat, "mode": mode}, None


# ------------------------------------------------------------ near duplicates
def build_groups(records, max_hamming, keep, rng, removed, cluster_rows):
    """Return list of groups (lists of records). Removes extra near-duplicates."""
    records = sorted(records, key=lambda r: r["rel"])
    hashable = [r for r in records if not r["flat"]]
    n = len(hashable)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        H = np.stack([r["dh"] for r in hashable])
        for i in range(n - 1):
            d = utils.POPCOUNT[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
            for j in np.nonzero(d <= max_hamming)[0]:
                ra, rb = find(i), find(i + 1 + int(j))
                if ra != rb:
                    parent[rb] = ra

    clusters = defaultdict(list)
    for i, r in enumerate(hashable):
        clusters[find(i)].append(r)

    groups = [[r] for r in records if r["flat"]]
    cid = 0
    for members in clusters.values():
        if len(members) == 1:
            groups.append(members)
            continue
        cid += 1
        classes = {m["cls"] for m in members}
        if len(classes) > 1:
            for m in members:
                removed.append({"rel": m["rel"], "cls": m["cls"],
                                "reason": "near_duplicate_conflicting_labels",
                                "detail": f"cluster {cid} spans {sorted(classes)}"})
            cluster_rows.append([cid, "+".join(sorted(classes)), len(members), 0,
                                 "|".join(m["rel"] for m in members)])
            continue
        members = sorted(members, key=lambda m: m["rel"])
        kept = members if len(members) <= keep else sorted(rng.sample(members, keep),
                                                           key=lambda m: m["rel"])
        kept_ids = {m["rel"] for m in kept}
        for m in members:
            if m["rel"] not in kept_ids:
                removed.append({"rel": m["rel"], "cls": m["cls"], "reason": "near_duplicate",
                                "detail": f"cluster {cid}"})
        cluster_rows.append([cid, members[0]["cls"], len(members), len(kept),
                             "|".join(m["rel"] for m in members)])
        groups.append(kept)
    return groups


# ------------------------------------------------------------------ split
def split_groups(groups, rng):
    """Stratified, group-aware assignment of one class. Returns {split: [records]}."""
    total = sum(len(g) for g in groups)
    targets = {s: total * config.SPLIT_RATIOS[s] for s in config.SPLIT_NAMES}
    counts = {s: 0 for s in config.SPLIT_NAMES}
    out = {s: [] for s in config.SPLIT_NAMES}
    order = sorted(groups, key=lambda g: g[0]["rel"])
    rng.shuffle(order)
    order.sort(key=len, reverse=True)       # stable: big groups first, random among equals
    for g in order:
        s = max(config.SPLIT_NAMES, key=lambda k: (targets[k] - counts[k]) / targets[k])
        out[s].extend(g)
        counts[s] += len(g)
    return out


def materialise(assign) -> list:
    for s in config.SPLIT_NAMES:
        for cls in config.CLASS_NAMES:
            d = config.SPLIT_DIR / s / cls
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    rows = []
    for (s, cls), recs in sorted(assign.items()):
        for r in sorted(recs, key=lambda x: x["rel"]):
            name = f"{r['sha'][:16]}.jpg"
            dest = config.SPLIT_DIR / s / cls / name
            utils.load_rgb(r["path"], max_side=config.STORE_MAX_SIDE).save(
                dest, "JPEG", quality=config.JPEG_QUALITY)
            rows.append([s, cls, name, r["rel"], r["sha"], r["gid"]])
    return rows


# ------------------------------------------------------------------ build
def build(max_hamming: int) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    found = scan_raw()
    before = {c: len(found[c]) for c in config.CLASS_NAMES}
    log.info("Counts before cleaning: %s", before)

    removed, converted = [], Counter()
    records = []
    total = sum(before.values())
    done = 0
    for cls in config.CLASS_NAMES:
        for path in found[cls]:
            rec, reason = inspect_image(path, cls)
            done += 1
            if done % 500 == 0:
                log.info("inspected %d / %d", done, total)
            if rec is None:
                removed.append({"rel": rel(path), "cls": cls, "reason": reason, "detail": ""})
            else:
                if rec["mode"] != "RGB":
                    converted[cls] += 1
                records.append(rec)

    # manual review removals
    manual = read_remove_list()
    known = {r["rel"] for r in records}
    for item in sorted(manual - known):
        log.warning("remove_list entry not found among usable images: %s", item)
    for r in records:
        if r["rel"] in manual:
            removed.append({"rel": r["rel"], "cls": r["cls"], "reason": "manual_review", "detail": ""})
    records = [r for r in records if r["rel"] not in manual]

    # exact duplicates, across ALL classes
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r["sha"]].append(r)
    drop = set()
    for grp in by_sha.values():
        if len(grp) == 1:
            continue
        grp.sort(key=lambda r: r["rel"])
        if len({r["cls"] for r in grp}) > 1:
            for r in grp:
                drop.add(r["rel"])
                removed.append({"rel": r["rel"], "cls": r["cls"],
                                "reason": "exact_duplicate_conflicting_labels", "detail": ""})
        else:
            for r in grp[1:]:
                drop.add(r["rel"])
                removed.append({"rel": r["rel"], "cls": r["cls"], "reason": "exact_duplicate",
                                "detail": f"same as {grp[0]['rel']}"})
    records = [r for r in records if r["rel"] not in drop]
    log.info("After exact-duplicate removal: %d images", len(records))

    # near duplicates (cross-class) -> groups
    rng = random.Random(config.SEED)
    cluster_rows = []
    groups_all = build_groups(records, max_hamming, config.NEAR_DUP_KEEP, rng, removed, cluster_rows)
    gid = 0
    for g in sorted(groups_all, key=lambda g: g[0]["rel"]):
        gid += 1
        for r in g:
            r["gid"] = f"g{gid:05d}"
    if cluster_rows:
        biggest = max(row[2] for row in cluster_rows)
        log.info("%d near-duplicate clusters, largest has %d images", len(cluster_rows), biggest)
        if biggest > 30:
            log.warning("A near-duplicate cluster has %d images. Open "
                        "reports/near_duplicate_clusters.csv and check it is not unrelated "
                        "photos; lower --near-dup-hamming if it is.", biggest)

    # split per class
    assign = {}
    after = Counter()
    for ci, cls in enumerate(config.CLASS_NAMES):
        cls_groups = [g for g in groups_all if g[0]["cls"] == cls]
        after[cls] = sum(len(g) for g in cls_groups)
        parts = split_groups(cls_groups, random.Random(config.SEED + ci))
        for s in config.SPLIT_NAMES:
            assign[(s, cls)] = parts[s]
            if not parts[s]:
                sys.exit(f"ERROR: class {cls} has no images in split {s}")
    log.info("Writing images to data/split ...")
    rows = materialise(assign)

    # verification
    seen = {}
    for s, cls, name, orig, sha, g in rows:
        if sha in seen and seen[sha] != s:
            sys.exit(f"ERROR: file hash {sha} appears in two splits")
        seen[sha] = s
    group_split = defaultdict(set)
    for s, cls, name, orig, sha, g in rows:
        group_split[g].add(s)
    if any(len(v) > 1 for v in group_split.values()):
        sys.exit("ERROR: a near-duplicate group spans two splits")
    on_disk = sum(len(utils.list_images(config.SPLIT_DIR / s)) for s in config.SPLIT_NAMES)
    if on_disk != len(rows):
        sys.exit(f"ERROR: manifest has {len(rows)} rows but {on_disk} files are on disk")

    # reports
    with open(config.REPORTS_DIR / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "filename", "original_path", "sha256", "group_id"])
        w.writerows(rows)
    with open(config.REPORTS_DIR / "removed_files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rel", "cls", "reason", "detail"])
        w.writeheader()
        w.writerows(sorted(removed, key=lambda x: x["rel"]))
    with open(config.REPORTS_DIR / "near_duplicate_clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "class", "size", "kept", "files"])
        w.writerows(cluster_rows)
    split_counts = {(s, c): len(assign[(s, c)]) for s in config.SPLIT_NAMES for c in config.CLASS_NAMES}
    write_summary(before, removed, converted, after, split_counts, len(manual), max_hamming)
    log.info("DONE. Final counts: %s", dict(after))
    log.info("Split sizes: %s", {s: sum(split_counts[(s, c)] for c in config.CLASS_NAMES)
                                 for s in config.SPLIT_NAMES})


def write_summary(before, removed, converted, after, split_counts, n_manual, max_hamming) -> None:
    reasons = ["unreadable", "empty_file", "too_small", "grayscale", "manual_review",
               "exact_duplicate", "exact_duplicate_conflicting_labels",
               "near_duplicate", "near_duplicate_conflicting_labels"]
    by = defaultdict(Counter)
    for r in removed:
        by[r["cls"]][r["reason"]] += 1
    L = []
    L.append("# Data summary\n")
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Seed {config.SEED}. "
             "Source: GVLiD (4 grape classes) + own Not_a_Grape_Leaf images. "
             "GVLiD metadata file not used.\n")
    L.append("## Counts before cleaning, removals, after cleaning\n")
    L.append("| Class | Before | " + " | ".join(reasons) + " | After |")
    L.append("|---|---|" + "---|" * len(reasons) + "---|")
    for c in config.CLASS_NAMES:
        L.append(f"| {c} | {before[c]} | " + " | ".join(str(by[c][r]) for r in reasons)
                 + f" | {after[c]} |")
    L.append(f"| **Total** | {sum(before.values())} | "
             + " | ".join(str(sum(by[c][r] for c in config.CLASS_NAMES)) for r in reasons)
             + f" | {sum(after.values())} |\n")
    L.append("## Split (stratified 70/15/15, seed 42, near-duplicate groups kept together)\n")
    L.append("| Class | Train | Val | Test | Total |")
    L.append("|---|---|---|---|---|")
    for c in config.CLASS_NAMES:
        t = [split_counts[(s, c)] for s in config.SPLIT_NAMES]
        L.append(f"| {c} | {t[0]} | {t[1]} | {t[2]} | {sum(t)} |")
    tot = [sum(split_counts[(s, c)] for c in config.CLASS_NAMES) for s in config.SPLIT_NAMES]
    L.append(f"| **Total** | {tot[0]} | {tot[1]} | {tot[2]} | {sum(tot)} |\n")
    L.append("## Notes\n")
    small = [c for c in config.CLASS_NAMES if after[c] < 800]
    if small:
        L.append(f"- Classes below the 800-original target after cleaning: {', '.join(small)}. "
                 "Class weights (computed from train counts) compensate.")
    else:
        L.append("- All classes have at least 800 images after cleaning.")
    L.append(f"- Manual review: {n_manual} files listed in reports/review/remove_list.txt were left out.")
    L.append(f"- Non-RGB images converted to RGB: {dict(converted) if converted else 'none'}.")
    L.append(f"- Near-duplicates: dhash distance <= {max_hamming}/64, at most "
             f"{config.NEAR_DUP_KEEP} kept per cluster. See reports/near_duplicate_clusters.csv.")
    L.append("- Every removed file and its reason: reports/removed_files.csv. "
             "Exact split list: reports/split_manifest.csv. data/raw is never modified.")
    L.append("- Images in data/split are re-saved as RGB JPEG (EXIF removed, longest side <= "
             f"{config.STORE_MAX_SIDE}px).")
    L.append("- Limitation: GVLiD comes from one phone on one date and no metadata is used for grouping, "
             "so photos from one session can still sit in different splits. "
             "Test scores can therefore look better than real use; the field test is the honest measure.")
    (config.REPORTS_DIR / "data_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    rv = sub.add_parser("review")
    rv.add_argument("--n", type=int, default=50)
    rv.add_argument("--seed", type=int, default=config.SEED)
    b = sub.add_parser("build")
    b.add_argument("--near-dup-hamming", type=int, default=config.NEAR_DUP_MAX_HAMMING)
    args = ap.parse_args()
    if args.cmd == "init":
        init_dirs()
    elif args.cmd == "review":
        make_review_sheets(args.n, args.seed)
    else:
        build(args.near_dup_hamming)


if __name__ == "__main__":
    main()