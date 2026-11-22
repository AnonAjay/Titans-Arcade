"""Tests for OneEuroFilter adaptive smoothing and pose tracking stability."""
import math
import numpy as np
import pytest
from types import SimpleNamespace

from pose_core.capture import OneEuroFilter, PoseTracker
from pose_core.landmarks import feet_info
from pose_core.floor_map import FloorMapper


def _make_dummy_landmarks(visible_ids: dict[int, tuple[float, float, float]]):
    lms = []
    for i in range(33):
        if i in visible_ids:
            x, y, v = visible_ids[i]
            lms.append(SimpleNamespace(x=x, y=y, visibility=v))
        else:
            lms.append(SimpleNamespace(x=0.0, y=0.0, visibility=0.0))
    return lms


def test_one_euro_filter_attenuates_stationary_noise():
    """At low speeds (standing still), the filter must heavily damp sensor jitter."""
    f = OneEuroFilter(min_cutoff=0.8, beta=0.02, d_cutoff=1.0)
    
    noisy_inputs = []
    filtered_outputs = []
    
    base_pos = np.array([300.0, 400.0])
    for frame in range(40):
        t = frame * 0.033  # ~30 FPS
        # High-frequency noise of +/- 5 pixels
        noise = 5.0 * math.sin(frame * 2.0)
        noisy_pos = base_pos + np.array([noise, -noise])
        noisy_inputs.append(noisy_pos[0])
        
        filtered = f.filter(noisy_pos, t)
        filtered_outputs.append(filtered[0])
        
    input_std = np.std(noisy_inputs[10:])
    output_std = np.std(filtered_outputs[10:])
    
    # Filtered signal variance must be substantially lower than input jitter
    assert output_std < input_std * 0.45


def test_one_euro_filter_tracks_fast_step_without_lag():
    """During fast movement, beta * speed must increase cutoff frequency, reducing lag."""
    f = OneEuroFilter(min_cutoff=0.8, beta=0.02, d_cutoff=1.0)
    
    # Standing still for 10 frames at x = 100
    for frame in range(10):
        f.filter(np.array([100.0, 200.0]), t=frame * 0.033)
        
    # Sudden fast step from 100 to 220 in 3 frames (120 px / 0.1s = 1200 px/s)
    t1 = 10 * 0.033
    t2 = 11 * 0.033
    t3 = 12 * 0.033
    
    f.filter(np.array([140.0, 200.0]), t=t1)
    f.filter(np.array([180.0, 200.0]), t=t2)
    p3 = f.filter(np.array([220.0, 200.0]), t=t3)
    
    # Should be close to target 220 (zero deadband lag)
    assert p3[0] > 195.0


def test_one_euro_filter_teleport_guard():
    """A jump greater than TELEPORT_PX should reset the filter immediately instead of smearing."""
    f = OneEuroFilter()
    f.filter(np.array([100.0, 100.0]), t=1.0)
    
    # Point teleported across screen (> 150 px)
    jumped = f.filter(np.array([400.0, 400.0]), t=1.033)
    assert jumped[0] == 400.0
    assert jumped[1] == 400.0


class DummyMapper:
    w_m = 2.0
    h_m = 2.0

    def to_floor(self, px, py):
        return (px * 0.005, py * 0.005)

    def inside(self, fx, fy):
        return 0.0 <= fx <= 2.0 and 0.0 <= fy <= 2.0


def test_pose_tracker_smooth_continuous_output():
    tracker = PoseTracker(DummyMapper())
    
    lms1 = _make_dummy_landmarks({
        27: (0.45, 0.85, 0.95), 28: (0.55, 0.85, 0.95),
        29: (0.44, 0.88, 0.95), 30: (0.54, 0.88, 0.95),
        31: (0.46, 0.92, 0.95), 32: (0.56, 0.92, 0.95),
        11: (0.45, 0.20, 0.95), 12: (0.55, 0.20, 0.95),
        23: (0.45, 0.50, 0.95), 24: (0.55, 0.50, 0.95),
    })
    
    pf1 = tracker.update([lms1], 640, 480, ts=1.0)
    assert pf1.tracked is True
    assert pf1.floor_x_m is not None
    assert pf1.left_foot_m is not None
    assert pf1.right_foot_m is not None


def test_floor_mapper_round_trips_projective_floor_points():
    """A perspective camera calibration should map floor corners and interior points both ways."""
    corners = [
        [82.0, 402.0],
        [562.0, 410.0],
        [490.0, 158.0],
        [164.0, 151.0],
    ]
    mapper = FloorMapper(corners, w_m=1.8, h_m=1.8)

    for floor_point in ((0.0, 0.0), (1.8, 0.0), (1.8, 1.8), (0.0, 1.8), (0.9, 0.9)):
        pixel_point = mapper.to_pixels(*floor_point)
        round_trip = mapper.to_floor(*pixel_point)
        assert round_trip == pytest.approx(floor_point, abs=1e-4)


@pytest.mark.xfail(
    strict=True,
    reason="known defect: collinear calibration corners produce a homography instead of rejection",
)
def test_floor_mapper_rejects_collinear_calibration_corners():
    """A common failed setup (three or more corners in a line) must fail clearly."""
    collinear_corners = [[50.0, 50.0], [100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]

    with pytest.raises(ValueError, match="Could not compute a homography"):
        FloorMapper(collinear_corners, w_m=1.8, h_m=1.8)
