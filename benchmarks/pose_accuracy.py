"""Evaluate the reference-image pose indexer against independent annotations.

The current catalog's landmarks were produced by MediaPipe and are not ground
truth. This tool refuses to use them as labels. Use --init to create an empty,
reviewed=false label template, then add independently reviewed coordinates.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_core.pose_matcher import extract_pose_angles
from pose_core.skeleton import BONES

JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_knee", "right_knee",
)
REQUIRED_POINT_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
ANGLE_TOLERANCE_DEG = 15.0
PCK_TORSO_FRACTION = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_label_template(catalog_path: Path, image_dir: Path, output_path: Path) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    poses = []
    for pose in catalog:
        image_path = image_dir / pose["image_file"]
        poses.append({
            "pose_id": pose["id"],
            "image_file": pose["image_file"],
            "image_sha256": _sha256(image_path) if image_path.is_file() else None,
            "reviewed": False,
            "points_norm": {},
        })
    template = {
        "schema_version": 1,
        "reviewed": False,
        "coordinate_convention": "Normalized source-image x,y in [0,1]. Keys are MediaPipe landmark indices; annotate independently of the saved catalog landmarks.",
        "required_point_ids": list(REQUIRED_POINT_IDS),
        "point_names": {
            "11": "left_shoulder", "12": "right_shoulder",
            "13": "left_elbow", "14": "right_elbow",
            "15": "left_wrist", "16": "right_wrist",
            "23": "left_hip", "24": "right_hip",
            "25": "left_knee", "26": "right_knee",
            "27": "left_ankle", "28": "right_ankle",
        },
        "angle_mapping": "Evaluator applies the same mirrored=true side mapping as the existing catalog indexer.",
        "poses": poses,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return len(poses)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def measure_detection_coverage(catalog_path: Path, image_dir: Path, model_path: Path, results_dir: Path) -> int:
    """Measure whether the current indexer emits six game angles; this is not accuracy."""
    import cv2
    from pose_core.blazer_indexer import BlazerIndexer

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    outputs: list[dict[str, object]] = []
    detected = 0
    complete = 0
    overlay_dir = results_dir / "pose_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    indexer = BlazerIndexer(model_path=model_path)
    try:
        for pose in catalog:
            image_path = image_dir / pose["image_file"]
            record: dict[str, object] = {
                "pose_id": pose["id"], "image_file": pose["image_file"],
                "image_sha256": _sha256(image_path) if image_path.is_file() else "",
                "detected": False, "complete_six_angles": False,
                "angles_json": "", "error": "",
            }
            image = cv2.imread(str(image_path)) if image_path.is_file() else None
            if image is None:
                record["error"] = "missing image" if not image_path.is_file() else "image decode failed"
            else:
                try:
                    angles, landmarks, _ = indexer.extract_from_image(image)
                    present = {joint: angles.get(joint) for joint in JOINTS}
                    record["detected"] = True
                    record["complete_six_angles"] = all(value is not None for value in present.values())
                    record["angles_json"] = json.dumps(present, sort_keys=True)
                    record["landmarks_norm_json"] = json.dumps(landmarks, separators=(",", ":"))
                    overlay = image.copy()
                    height, width = image.shape[:2]
                    core_ids = set(REQUIRED_POINT_IDS)
                    for first, second in BONES:
                        p1 = landmarks[first]
                        p2 = landmarks[second]
                        x1, y1 = int(p1[0] * width), int(p1[1] * height)
                        x2, y2 = int(p2[0] * width), int(p2[1] * height)
                        if -width <= x1 <= 2 * width and -height <= y1 <= 2 * height and -width <= x2 <= 2 * width and -height <= y2 <= 2 * height:
                            cv2.line(overlay, (x1, y1), (x2, y2), (255, 190, 0), max(1, width // 250), cv2.LINE_AA)
                    for point_id, point in enumerate(landmarks):
                        x, y = int(point[0] * width), int(point[1] * height)
                        color = (40, 220, 80) if point_id in core_ids else (0, 160, 255)
                        if -width <= x <= 2 * width and -height <= y <= 2 * height:
                            cv2.circle(overlay, (x, y), max(2, width // 100), color, -1, cv2.LINE_AA)
                    overlay_path = overlay_dir / f"{pose['id']}.jpg"
                    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    record["overlay_path"] = overlay_path.relative_to(ROOT).as_posix()
                    overlay_pairs.append((pose["id"], image, overlay))
                    detected += 1
                    complete += int(bool(record["complete_six_angles"]))
                except ValueError as exc:
                    record["error"] = str(exc)
            outputs.append(record)
    finally:
        indexer.close()

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "pose_detection_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["pose_id", "image_file", "image_sha256", "detected", "complete_six_angles", "angles_json", "landmarks_norm_json", "overlay_path", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outputs)
    sheet_dir = results_dir / "pose_visual_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = 4
    panel_w, panel_h, label_h = 360, 450, 26
    for sheet_idx in range((len(overlay_pairs) + per_sheet - 1) // per_sheet):
        group = overlay_pairs[sheet_idx * per_sheet:(sheet_idx + 1) * per_sheet]
        sheet = np.full((2 * (panel_h + label_h), 2 * 2 * panel_w, 3), 245, dtype=np.uint8)
        for idx, (pose_id, original, overlay) in enumerate(group):
            row, col = divmod(idx, 2)
            y0 = row * (panel_h + label_h)
            x0 = col * 2 * panel_w
            for image_idx, image_part, label in ((0, original, "SOURCE"), (1, overlay, "MEDIAPIPE DOTS")):
                resized = cv2.resize(image_part, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                px = x0 + image_idx * panel_w
                sheet[y0 + label_h:y0 + label_h + panel_h, px:px + panel_w] = resized
                cv2.putText(sheet, f"{pose_id} | {label}", (px + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.imwrite(str(sheet_dir / f"pairs-{sheet_idx + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Do not erase a completed audit when refreshing overlays. Save an empty
    # review template only on first run; later runs update images/raw outputs.
    audit_path = results_dir / "visual_pose_audit.json"
    if not audit_path.exists():
        audit_template = {
            "reviewer": "ChatGPT vision audit, single reviewer",
            "reviewed": False,
            "rubric": {
                "visually_plausible": "The six game-joint neighborhoods appear to correspond to visible body parts.",
                "uncertain": "Occlusion, cropping, scale, stylization, or viewpoint makes visual judgment unreliable.",
                "obvious_mismatch": "At least one required joint is clearly placed off its corresponding body part.",
            },
            "poses": [{"pose_id": pose["pose_id"], "image_file": pose["image_file"], "rating": None, "notes": ""} for pose in outputs],
            "summary": {
                "images": 0, "visually_plausible": 0, "uncertain": 0,
                "obvious_mismatch": 0,
                "wilson_95_ci": {"visually_plausible": [0.0, 0.0]},
            },
            "limitation": "AI visual plausibility is not independent ground truth or an accuracy estimate. Keep reviewed=false until a human reviews the judgments.",
        }
        audit_path.write_text(json.dumps(audit_template, indent=2) + "\n", encoding="utf-8")
    detect_ci = wilson_interval(detected, len(catalog))
    complete_ci = wilson_interval(complete, len(catalog))
    report = [
        "MEDIAPIPE REFERENCE-IMAGE OUTPUT COVERAGE (NOT ACCURACY)",
        f"Catalog images = {len(catalog)}; model detected = {detected}; detection yield = {detected / len(catalog):.4f} (95% Wilson CI {detect_ci[0]:.4f}..{detect_ci[1]:.4f})",
        f"Images with all six game angles = {complete}; yield = {complete / len(catalog):.4f} (95% Wilson CI {complete_ci[0]:.4f}..{complete_ci[1]:.4f})",
        "Caveat = BlazerIndexer currently drops landmark visibility before angle extraction; this output-yield count does not mean joints were visibly supported.",
        f"Visual comparison sheets = {sheet_dir.relative_to(ROOT).as_posix() if sheet_dir.is_relative_to(ROOT) else sheet_dir}",
        "Visual audit labels = reviewed=false; AI visual judgment is not ground truth.",
        "Pose correctness = NOT MEASURED; no independent reviewed labels are available.",
        f"Raw per-image outputs = {raw_path.relative_to(ROOT).as_posix() if raw_path.is_relative_to(ROOT) else raw_path}",
        "Production catalog was not written.",
    ]
    (results_dir / "POSE_DETECTION.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


def _reference_angles(points_norm: dict[str, list[float]]) -> dict[str, float | None]:
    landmarks: list[list[float] | None] = [None] * 33
    for point_id, xy in points_norm.items():
        landmarks[int(point_id)] = [float(xy[0]), float(xy[1])]
    return extract_pose_angles(landmarks, mirrored=True)


def _torso_scale(points_norm: dict[str, list[float]]) -> float:
    left_shoulder = np.asarray(points_norm["11"], dtype=float)
    right_shoulder = np.asarray(points_norm["12"], dtype=float)
    left_hip = np.asarray(points_norm["23"], dtype=float)
    right_hip = np.asarray(points_norm["24"], dtype=float)
    shoulders_mid = (left_shoulder + right_shoulder) / 2.0
    hips_mid = (left_hip + right_hip) / 2.0
    return float(np.linalg.norm(shoulders_mid - hips_mid))


def evaluate(labels_path: Path, image_dir: Path, model_path: Path, results_dir: Path) -> int:
    import cv2
    from pose_core.blazer_indexer import BlazerIndexer

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    pose_labels = labels.get("poses", [])
    labeled_poses = []
    incomplete = []
    for label in pose_labels:
        missing = [str(point_id) for point_id in REQUIRED_POINT_IDS if str(point_id) not in label.get("points_norm", {})]
        if missing:
            incomplete.append({"pose_id": label.get("pose_id"), "missing_point_ids": missing})
        else:
            labeled_poses.append(label)

    report_path = results_dir / "POSE_ACCURACY.txt"
    if not labeled_poses:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [
            "POSE ACCURACY: NOT MEASURED",
            f"Labels file = {labels_path.relative_to(ROOT).as_posix() if labels_path.is_relative_to(ROOT) else labels_path}",
            f"Images with complete labels = 0 / {len(pose_labels)}",
            f"Images missing required landmarks = {len(incomplete)}",
            "Reason = independent landmark annotations are absent or incomplete; saved catalog landmarks are MediaPipe outputs and are not accepted as ground truth.",
            "Required points per pose = " + ", ".join(map(str, REQUIRED_POINT_IDS)),
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        print("Accuracy NOT MEASURED: independent labels are absent or incomplete.")
        return 2

    rows: list[dict[str, object]] = []
    coord_errors: list[float] = []
    angle_errors: list[float] = []
    angle_successes = 0
    angle_total = 0
    pose_all_joint_successes = 0
    indexed_images = 0
    indexer = BlazerIndexer(model_path=model_path)
    try:
        for label in labeled_poses:
            image_path = image_dir / label["image_file"]
            expected_hash = label.get("image_sha256")
            actual_hash = _sha256(image_path)
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"Image changed since annotation: {image_path}; regenerate/review labels.")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Image decode failed: {image_path}")
            try:
                predicted_angles, predicted_landmarks, _ = indexer.extract_from_image(image)
                indexed_images += 1
                reference_angles = _reference_angles(label["points_norm"])
                torso_scale = _torso_scale(label["points_norm"])
                pose_joint_ok = True
                for point_id in REQUIRED_POINT_IDS:
                    point_key = str(point_id)
                    expected_xy = np.asarray(label["points_norm"][point_key], dtype=float)
                    predicted_xy = np.asarray(predicted_landmarks[point_id], dtype=float)
                    error = float(np.linalg.norm(predicted_xy - expected_xy))
                    normalized_error = error / torso_scale if torso_scale > 0 else float("inf")
                    coord_errors.append(normalized_error)
                    rows.append({
                        "pose_id": label["pose_id"], "image_file": label["image_file"],
                        "sample_type": "keypoint", "joint": point_key,
                        "expected": json.dumps(expected_xy.tolist()),
                        "predicted": json.dumps(predicted_xy.tolist()),
                        "error": f"{normalized_error:.8f}", "unit": "torso_lengths",
                    })
                for joint in JOINTS:
                    expected_angle = reference_angles.get(joint)
                    predicted_angle = predicted_angles.get(joint)
                    if expected_angle is None or predicted_angle is None:
                        pose_joint_ok = False
                        rows.append({
                            "pose_id": label["pose_id"], "image_file": label["image_file"],
                            "sample_type": "angle", "joint": joint,
                            "expected": expected_angle, "predicted": predicted_angle,
                            "error": "", "unit": "degrees", "failure": "angle unavailable",
                        })
                        continue
                    error_deg = abs(float(predicted_angle) - float(expected_angle))
                    angle_errors.append(error_deg)
                    angle_total += 1
                    success = error_deg <= ANGLE_TOLERANCE_DEG
                    angle_successes += int(success)
                    pose_joint_ok = pose_joint_ok and success
                    rows.append({
                        "pose_id": label["pose_id"], "image_file": label["image_file"],
                        "sample_type": "angle", "joint": joint,
                        "expected": f"{float(expected_angle):.4f}",
                        "predicted": f"{float(predicted_angle):.4f}",
                        "error": f"{error_deg:.4f}", "unit": "degrees",
                    })
                pose_all_joint_successes += int(pose_joint_ok)
            except ValueError as exc:
                rows.append({
                    "pose_id": label["pose_id"], "image_file": label["image_file"],
                    "sample_type": "image", "joint": "", "expected": "pose detected",
                    "predicted": "not detected", "error": "", "unit": "", "failure": str(exc),
                })
    finally:
        indexer.close()

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "pose_accuracy_raw.csv"
    fields = ["pose_id", "image_file", "sample_type", "joint", "expected", "predicted", "error", "unit", "failure"]
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    low, high = wilson_interval(angle_successes, angle_total)
    reviewed = bool(labels.get("reviewed")) and all(bool(pose.get("reviewed")) for pose in labeled_poses)
    status = "MEASURED" if reviewed else "PROVISIONAL"
    pose_low, pose_high = wilson_interval(pose_all_joint_successes, len(labeled_poses))
    report = [
        "POSE ACCURACY RESULTS",
        f"Dataset reviewed = {str(reviewed).lower()}",
        f"Images with complete labels = {len(labeled_poses)} / {len(pose_labels)}; incomplete/unlabeled = {len(incomplete)}",
        f"Detected = {indexed_images} / {len(labeled_poses)}; detection coverage = {indexed_images / len(labeled_poses):.4f}",
        f"Angle criterion = absolute error <= {ANGLE_TOLERANCE_DEG:.1f} degrees",
        f"metric = six-joint angle pass rate {angle_successes / angle_total if angle_total else 0.0:.4f} proportion | n = {angle_total} joint samples | 95% CI = Wilson [{low:.4f}, {high:.4f}] | status = {status}",
        f"metric = all-six-joints image pass rate {pose_all_joint_successes / len(labeled_poses):.4f} proportion | n = {len(labeled_poses)} images | 95% CI = Wilson [{pose_low:.4f}, {pose_high:.4f}] | status = {status}",
        f"metric = keypoint normalized error median {statistics.median(coord_errors) if coord_errors else 0.0:.6f} torso_lengths | n = {len(coord_errors)} joint samples | status = {status}",
        f"metric = joint angle error median {statistics.median(angle_errors) if angle_errors else 0.0:.4f} degrees | n = {len(angle_errors)} joint samples | status = {status}",
        f"Raw samples = {raw_path.relative_to(ROOT).as_posix()}",
        "Existing production catalog was read only; no catalog updates were made.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="create an empty reviewed=false label template")
    parser.add_argument("--coverage-only", action="store_true", help="measure detection/angle-output yield, not correctness")
    parser.add_argument("--labels", type=Path, default=ROOT / "benchmarks" / "pose_labels.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "poses_index.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "poses")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_lite.task")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()
    if args.init:
        count = create_label_template(args.catalog, args.images, args.labels)
        print(f"Created {args.labels} with {count} reviewed=false image records.")
        return 0
    if not args.labels.is_file():
        print(f"Labels not found: {args.labels}. Run with --init to create the annotation template.")
        return 2
    if not args.model.is_file():
        print(f"Model not found: {args.model}. Run python -m tools.get_models first.")
        return 2
    if args.coverage_only:
        return measure_detection_coverage(args.catalog, args.images, args.model, args.results)
    return evaluate(args.labels, args.images, args.model, args.results)


if __name__ == "__main__":
    raise SystemExit(main())
