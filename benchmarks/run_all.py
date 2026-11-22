"""Run deterministic, hardware-independent Titans-Arcade microbenchmarks.

This measures existing production functions without changing their behavior.
Every warm-up and timed sample, including its output, is written to a timestamped CSV.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable, Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
sys.path.insert(0, str(ROOT))

from pose_core.blazer_matcher import BlazerPoseTemplate, evaluate_blazer_pose
from pose_core.floor_map import FloorMapper
from shared.contracts import Phase, PoseFrame
from station_a.logic import GRID_SIZE, LavaEngine, LavaWave, TournamentPhase, WavePattern


def _floor_mapping_case() -> tuple[Callable[[], Any], dict[str, Any]]:
    mapper = FloorMapper(
        [[82.0, 402.0], [562.0, 410.0], [490.0, 158.0], [164.0, 151.0]],
        w_m=1.8,
        h_m=1.8,
    )
    pixel_point = (322.0, 280.0)

    def run() -> dict[str, Any]:
        x_m, y_m = mapper.to_floor(*pixel_point)
        return {"pixel": list(pixel_point), "floor_m": [x_m, y_m]}

    return run, {"pixel": list(pixel_point), "corners_px": mapper.corners_px.tolist()}


def _lava_update_case() -> tuple[Callable[[], Any], dict[str, Any]]:
    mask = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    for col in range(GRID_SIZE):
        for row in range(GRID_SIZE):
            mask[col][row] = (col + row) % 3 == 0
    pose = PoseFrame(tracked=True, floor_x_m=0.99, floor_y_m=0.99, airborne=False)
    engine = LavaEngine(duration_s=30.0)
    engine.start_game(player_id="P009", player_name="Benchmark Player")
    engine.tournament_phase = TournamentPhase.ROUND_ACTIVE

    def run() -> dict[str, Any]:
        engine.wave = LavaWave(
            pattern=WavePattern.CLUSTER,
            timer=2.0,
            total_warn=2.0,
            target_mask=[row[:] for row in mask],
            heat=[[0.0] * GRID_SIZE for _ in range(GRID_SIZE)],
        )
        engine.state.phase = Phase.PLAYING
        engine.state.score = 0
        engine.state.hits = 0
        engine.state.dodges = 0
        engine.state.streak = 0
        state, wave = engine.update(pose, dt=1.0 / 60.0)
        return {
            "phase": state.phase.value,
            "hits": state.hits,
            "dodges": state.dodges,
            "timer_s": wave.timer if wave else None,
        }

    workload = {
        "player_id": "P009",
        "pose": {"tracked": True, "floor_x_m": 0.99, "floor_y_m": 0.99, "airborne": False},
        "wave_mask": mask,
        "dt_s": 1.0 / 60.0,
    }
    return run, workload


def _blazer_match_case() -> tuple[Callable[[], Any], dict[str, Any]]:
    target_angles = {
        "left_shoulder": 90.0,
        "right_shoulder": 90.0,
        "left_elbow": 165.0,
        "right_elbow": 165.0,
        "left_knee": 170.0,
        "right_knee": 170.0,
    }
    pose_angles = {
        "left_shoulder": 93.0,
        "right_shoulder": 87.0,
        "left_elbow": 161.0,
        "right_elbow": 168.0,
        "left_knee": 166.0,
        "right_knee": 173.0,
    }
    template = BlazerPoseTemplate(
        id="benchmark-t-pose",
        name="Benchmark T Pose",
        target_angles=target_angles,
        key_joints=["left_shoulder", "right_shoulder"],
    )

    def run() -> dict[str, Any]:
        result = evaluate_blazer_pose(pose_angles, template, active_joints=tuple(target_angles))
        return {"match_pct": result.match_pct, "is_matched": result.is_matched, "missing": result.missing_joints}

    return run, {"pose_angles_deg": pose_angles, "target_angles_deg": target_angles, "head_check": True}


def _hardware_environment() -> str:
    cpu = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown CPU")
    return f"CPU: {cpu}; OpenCV: {cv2.__version__}; NumPy: {np.__version__}; camera: not accessed"


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _format_metric(name: str, values_ms: list[float], hardware: str, script: str, n_min: int) -> str:
    ordered = sorted(values_ms)
    q1 = _percentile(ordered, 25)
    q3 = _percentile(ordered, 75)
    p95 = _percentile(ordered, 95) if len(ordered) >= 20 else None
    parts = [f"median={statistics.median(ordered):.6f} ms", f"IQR={q1:.6f}..{q3:.6f} ms",
             f"min={min(ordered):.6f} ms", f"max={max(ordered):.6f} ms"]
    if p95 is not None:
        parts.append(f"p95={p95:.6f} ms")
    status = "MEASURED" if len(ordered) >= n_min else "INDICATIVE ONLY"
    return (
        f"metric = {name} median {statistics.median(ordered):.6f} ms | baseline = none | n = {len(ordered)}"
        f" | spread = {', '.join(parts)}"
        f" | hardware = {hardware} | script = {script} | status = {status}"
    )


def _run_case(
    name: str,
    workload: dict[str, Any],
    operation: Callable[[], Any],
    warmups: int,
    runs: int,
    writer: csv.DictWriter,
) -> tuple[list[float], list[float]]:
    timings_ms: list[float] = []
    warmup_ms: list[float] = []
    for kind, count in (("warmup", warmups), ("timed", runs)):
        for index in range(count):
            start = time.perf_counter()
            output = operation()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            writer.writerow({
                "case": name,
                "sample_type": kind,
                "sample_index": index,
                "elapsed_ms": f"{elapsed_ms:.9f}",
                "input_json": json.dumps(workload, sort_keys=True, separators=(",", ":")),
                "output_json": json.dumps(output, sort_keys=True, separators=(",", ":")),
            })
            if kind == "timed":
                timings_ms.append(elapsed_ms)
            else:
                warmup_ms.append(elapsed_ms)
    return timings_ms, warmup_ms


def _environment_block(commit: str) -> str:
    return "\n".join([
        "ENVIRONMENT",
        f"CPU = {os.environ.get('PROCESSOR_IDENTIFIER', platform.processor() or 'unknown')}",
        "GPU/VRAM = not measured",
        "RAM = not measured",
        f"OS = {platform.platform()}",
        f"Python = {platform.python_version()}",
        f"NumPy = {np.__version__}",
        f"OpenCV = {cv2.__version__}",
        "Model = not used; quantization = not applicable",
        f"Git commit = {commit}",
        f"Date = {dt.datetime.now(dt.timezone.utc).isoformat()}",
    ])


def _pose_coverage_metric() -> str:
    report_path = RESULTS / "POSE_DETECTION.txt"
    if not report_path.is_file():
        return (
            "metric = MediaPipe six-angle output yield not measured | baseline = none | n = 0 images"
            " | spread = model coverage run not available | hardware = not measured"
            " | script = benchmarks/pose_accuracy.py | status = SKIPPED_NO_BASELINE"
        )
    report_text = report_path.read_text(encoding="utf-8")
    match = re.search(
        r"Images with all six game angles = (\d+); yield = ([0-9.]+) \(95% Wilson CI ([0-9.]+)\.\.([0-9.]+)\)",
        report_text,
    )
    count_match = re.search(r"Catalog images = (\d+); model detected = (\d+)", report_text)
    if not match or not count_match:
        return (
            "metric = MediaPipe six-angle output yield not measured | baseline = none | n = 0 images"
            " | spread = could not parse pose coverage report | hardware = not measured"
            " | script = benchmarks/pose_accuracy.py | status = FAILED"
        )
    success, rate, low, high = match.groups()
    total, _ = count_match.groups()
    return (
        f"metric = MediaPipe six-angle output yield {success}/{total} images ({rate})"
        f" | baseline = none | n = {total} images | spread = 95% Wilson CI [{low}, {high}]"
        " | hardware = local CPU via MediaPipe XNNPACK"
        " | script = benchmarks/pose_accuracy.py | status = MEASURED"
    )


def _visual_audit_metric() -> str:
    audit_path = RESULTS / "visual_pose_audit.json"
    if not audit_path.is_file():
        return (
            "metric = AI visual alignment audit not run | baseline = none | n = 0 images"
            " | spread = no visual ratings | hardware = not applicable"
            " | script = benchmarks/pose_accuracy.py | status = SKIPPED_NO_BASELINE"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rated = [pose for pose in audit.get("poses", []) if pose.get("rating")]
    if not rated:
        return (
            "metric = AI visual alignment audit template only | baseline = none | n = 0 rated images"
            " | spread = no visual ratings | hardware = not applicable"
            " | script = benchmarks/pose_accuracy.py | status = SKIPPED_NO_BASELINE"
        )
    summary = audit["summary"]
    total = summary["images"]
    plausible = summary["visually_plausible"]
    uncertain = summary["uncertain"]
    mismatch = summary["obvious_mismatch"]
    ci = summary["wilson_95_ci"]["visually_plausible"]
    return (
        f"metric = ChatGPT visual plausibility ratings {plausible}/{total} plausible, {uncertain} uncertain, {mismatch} obvious mismatches"
        f" | baseline = none | n = {total} images | spread = 95% Wilson CI for plausible-label rate [{ci[0]:.4f}, {ci[1]:.4f}]; single reviewer"
        " | hardware = multimodal review of local source/overlay sheets"
        " | script = benchmarks/pose_accuracy.py | status = PROVISIONAL"
    )


def _run_tests() -> tuple[str, str]:
    per_run_temp = Path(tempfile.mkdtemp(prefix="titans-arcade-pytest-"))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-o", "addopts=", "--basetemp", str(per_run_temp)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (RESULTS / "pytest.txt").write_text(proc.stdout, encoding="utf-8")
    summary = next((line.strip() for line in reversed(proc.stdout.splitlines()) if " passed" in line or " failed" in line), "summary unavailable")
    state = "PASS" if proc.returncode == 0 else "FAIL"
    return state, summary


def _run_failure_case() -> dict[str, Any]:
    from shared.station_client import StationNetworkClient

    outbox_case = json.loads((ROOT / "benchmarks" / "cases" / "outbox_malformed.json").read_text(encoding="utf-8"))
    outbox = RESULTS / "malformed_outbox_case.jsonl"
    outbox.write_text(outbox_case["input_line"] + "\n", encoding="utf-8")
    client = StationNetworkClient(station="A", server_url="http://testserver", outbox_path=outbox)
    client._http_request = lambda method, endpoint, data=None: None
    client._sync_outbox_locked()
    actual = outbox.read_text(encoding="utf-8")
    outbox_result = {
        "case_id": outbox_case["case_id"],
        "reviewed": outbox_case["reviewed"],
        "input_line": outbox_case["input_line"],
        "expected_remaining_line": outbox_case["expected_remaining_line"],
        "actual_outbox": actual,
        "passed": actual.rstrip("\n") == outbox_case["expected_remaining_line"],
    }
    floor_case = json.loads((ROOT / "benchmarks" / "cases" / "floor_collinear.json").read_text(encoding="utf-8"))
    try:
        mapper = FloorMapper(floor_case["corners_px"], floor_case["floor_width_m"], floor_case["floor_height_m"])
        observed = {"accepted": True, "homography": mapper.H.tolist()}
    except ValueError as exc:
        observed = {"accepted": False, "error": str(exc)}
    floor_result = {
        "case_id": floor_case["case_id"],
        "reviewed": floor_case["reviewed"],
        "input_corners_px": floor_case["corners_px"],
        "expected": floor_case["expected"],
        "observed": observed,
        "passed": not observed["accepted"],
    }
    return {"cases": [outbox_result, floor_result]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="timed samples per operation (minimum 30)")
    parser.add_argument("--warmups", type=int, default=3, help="discarded warm-up samples per operation (minimum 3)")
    args = parser.parse_args()
    if args.runs < 30 or args.warmups < 3:
        parser.error("--runs must be >=30 and --warmups must be >=3 to match the benchmark protocol")

    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS / "raw" / f"samples-{timestamp}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    git_result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    commit = git_result.stdout.strip() if git_result.returncode == 0 else "unavailable"

    cases = [
        ("floor_mapping_one_point", *_floor_mapping_case()),
        ("station_a_lava_update_10x10", *_lava_update_case()),
        ("station_b_pose_match_six_joints", *_blazer_match_case()),
    ]
    metrics: list[str] = []
    hardware = _hardware_environment()
    with raw_path.open("w", encoding="utf-8", newline="") as sample_file:
        fields = ["case", "sample_type", "sample_index", "elapsed_ms", "input_json", "output_json"]
        writer = csv.DictWriter(sample_file, fieldnames=fields)
        writer.writeheader()
        for name, operation, workload in cases:
            samples, warmups = _run_case(name, workload, operation, args.warmups, args.runs, writer)
            metrics.append(_format_metric(name, samples, hardware, "benchmarks/run_all.py", 30))
            metrics.append(_format_metric(f"{name} warm-up samples (indicative only)", warmups, hardware, "benchmarks/run_all.py", 3))

    test_state, test_summary = _run_tests()
    suite_numbers = {part.split()[1]: int(part.split()[0]) for part in test_summary.split(",") if len(part.split()) >= 2 and part.split()[0].isdigit()}
    passed = suite_numbers.get("passed", 0)
    failed = suite_numbers.get("failed", 0)
    xfailed = suite_numbers.get("xfailed", 0)
    skipped = suite_numbers.get("skipped", 0)
    total = passed + failed + xfailed + skipped
    metrics.append(
        f"metric = pytest suite outcomes {passed} passed, {failed} failed, {xfailed} xfailed, {skipped} skipped"
        f" | baseline = none | n = {total} tests | spread = not applicable"
        f" | hardware = {platform.platform()} | script = pytest.ini | status = {'MEASURED' if test_state == 'PASS' else 'FAILED'}"
    )

    failure_case = _run_failure_case()
    failure_path = RESULTS / "failure-case.json"
    failure_path.write_text(json.dumps(failure_case, indent=2) + "\n", encoding="utf-8")
    for result in failure_case["cases"]:
        metrics.append(
            f"metric = {result['case_id']} reproduction {'pass' if result['passed'] else 'fail'} (1 case)"
            f" | baseline = none | n = 1 case | spread = indicative only"
            f" | hardware = local filesystem and CPU"
            f" | script = benchmarks/run_all.py | status = {'PROVISIONAL' if not result['reviewed'] else ('MEASURED' if result['passed'] else 'FAILED')}"
        )

    coverage_status = "pytest-cov available" if importlib.util.find_spec("pytest_cov") else "not measured (pytest-cov is not pinned in requirements.txt)"
    clone_report_path = RESULTS / "fresh-clone.txt"
    clone_report = clone_report_path.read_text(encoding="utf-8").strip() if clone_report_path.exists() else "Not run."
    catalog_count: int | None = None
    try:
        catalog_data = json.loads((ROOT / "data" / "poses_index.json").read_text(encoding="utf-8"))
        catalog_count = len(catalog_data) if isinstance(catalog_data, list) else len(catalog_data.get("poses", []))
    except Exception:
        pass
    catalog_claim = (
        f"- README: '36 ... poses' — SUPPORTED as {catalog_count} catalog entries in data/poses_index.json; this does not measure matching accuracy."
        if catalog_count == 36 else
        f"- README: '36 ... poses' — CONTRADICTED by {catalog_count} catalog entries in data/poses_index.json; this does not measure matching accuracy."
        if catalog_count is not None else
        "- README: '36 ... poses' — UNTESTABLE; catalog could not be read."
    )
    evidence = [
        _environment_block(commit),
        "",
        "MEASURED METRICS",
        *metrics,
        _pose_coverage_metric(),
        _visual_audit_metric(),
        "metric = MediaPipe pose correctness | baseline = none | n = 0 independently labeled images | spread = not available | hardware = not applicable | script = benchmarks/pose_accuracy.py | status = SKIPPED_NO_BASELINE",
        "",
        "FRESH-CLONE REPRODUCIBILITY",
        clone_report,
        "",
        "CLAIMS AUDIT",
        "- README: '<15 ms latency' and '60 FPS fluidity' — UNTESTABLE here; no camera was accessed and no frame pipeline was measured.",
        "- README: 'sub-millimeter stability' — UNTESTABLE here; no floor calibration accuracy dataset or physical ground-truth measurements are included.",
        "- README: 'sub-millisecond REST & SSE backend' — UNTESTABLE from the logic microbenchmarks; the suite exercises endpoints functionally but does not measure a running server under load.",
        "- README: '60 FPS real-time big-screen leaderboard', 30 FPS vision loop, and 60 FPS game loop — UNTESTABLE; no live capture or renderer was measured.",
        "- README: 'Zero cloud latency' and '100% offline edge vision' — the latency phrase is not a measurement; no runtime network audit was run.",
        "- README: '1st Place Winner' — untestable from repository evidence; event records are not present.",
        catalog_claim,
        "",
        "NOT MEASURED",
        "- Webcam capture/inference/render latency, FPS, CPU/GPU utilization: camera was not accessed; no reviewed frame/clip dataset is supplied.",
        "- MediaPipe reference-pose correctness: NOT MEASURED; the catalog's saved landmarks/angles come from MediaPipe and no independent annotations exist. See benchmarks/pose_accuracy.py and benchmarks/results/POSE_ACCURACY.txt.",
        "- Landmark visibility reliability: NOT MEASURED; BlazerIndexer discards visibility values before extracting catalog angles, so numeric output yield is not a visibility or correctness score.",
        "- CI: no CI configuration file or workflow was found in this repository.",
        f"- Coverage: {coverage_status}.",
        "- Process cold-start time, full-process memory, REST concurrency, and hardware round-trip latency: not measured by this logic microbenchmark. Three function warm-ups per path are logged separately.",
        "",
        "LIMITATIONS",
        "- Single Windows machine; CPU logic and OpenCV homography timings are local microbenchmarks, not end-to-end station latency.",
        "- Benchmark inputs are author-written deterministic fixtures; reviewed=false, so any correctness interpretation is PROVISIONAL.",
        "- No A/B optimization or baseline comparison was run; baseline = none.",
        "- The malformed-outbox and collinear-floor cases expose observed failure paths using author-written inputs; both are unreviewed.",
        "",
        "TOP 5 RESULTS WORTH PUTTING ON A RESUME",
        "- No results selected: the timings have no comparison baseline and only n=30 samples per path on one local machine.",
        "",
        f"Artifacts: raw samples={raw_path.relative_to(ROOT).as_posix()}; pytest={ (RESULTS / 'pytest.txt').relative_to(ROOT).as_posix() }; failure case={failure_path.relative_to(ROOT).as_posix()}; fresh clone={clone_report_path.relative_to(ROOT).as_posix()}.",
    ]
    (RESULTS / "EVIDENCE.txt").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    print(f"Raw samples: {raw_path.relative_to(ROOT)}")
    print(f"Evidence: {(RESULTS / 'EVIDENCE.txt').relative_to(ROOT)}")
    print(f"Test suite: {test_summary}")
    print("Failure reproductions: " + ", ".join(f"{item['case_id']}={'pass' if item['passed'] else 'fail'}" for item in failure_case["cases"]))
    return 0 if test_state == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
