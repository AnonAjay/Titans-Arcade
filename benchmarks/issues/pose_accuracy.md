# Add an independent MediaPipe pose accuracy benchmark

## Problem

The pose catalog currently stores `landmarks_norm` and `target_angles` generated
by MediaPipe. Comparing a fresh MediaPipe run to those values is circular; it
measures repeatability, not correctness. The current repository has no reviewed
ground-truth joint labels for its reference images, so it cannot support an
accuracy percentage or a claim that MediaPipe produced the correct pose.
The current indexer also converts detected landmarks to x/y coordinates and
drops visibility before angle extraction, so a complete numeric angle record
does not establish that its joints were visually supported.

The existing gameplay pipeline should remain unchanged while this work is
developed.

## Proposed scope

- Add an independently reviewed annotation set for a representative sample of
  the bundled pose images. Do not copy MediaPipe predictions into the labels.
- Record image hashes, annotation convention, reviewer state, and the relevant
  shoulder, elbow, hip, knee, and ankle points.
- Preserve per-joint visibility in the experimental report and do not treat it
  as calibrated correctness confidence without an evaluation.
- Compare the current MediaPipe indexer against these labels. Report angle error
  per game joint, normalized keypoint error, per-image all-joints pass rate,
  detection coverage, and failures.
- Save raw per-image/per-joint samples and report sample counts plus confidence
  intervals. Mark all outputs `PROVISIONAL` until labels are reviewed.
- Keep production `data/poses_index.json` read-only. Any experimental output must
  use a separate file and an explicit opt-in command.
- Add difficult examples: stylized art, occlusion, side views, cropped limbs,
  mirrored/ambiguous views, and single versus multiple people.

## Acceptance criteria

- Accuracy is evaluated against independent reviewed labels, never the current
  MediaPipe-generated catalog fields.
- The report states its angular tolerance and keypoint normalization rule.
- Coverage and rejection rate are reported with accuracy to expose selective
  rejection.
- Raw results are reproducible from one documented command, and image changes
  invalidate their corresponding labels.
- No production path or default pose catalog changes as part of this issue.

## Current evidence

`benchmarks/pose_accuracy.py` and its empty `reviewed=false` annotation template
provide the evaluation scaffold. Current status: pose accuracy is not measured
because no independent labels are available. No claim above 90% is supported.

The current indexer emitted all six game angles for 36/36 bundled images (95%
Wilson interval for output yield: 90.36%–100%). This is detection/output
coverage, not an accuracy score. Raw outputs are in
`benchmarks/results/pose_detection_raw.csv`.

A single ChatGPT visual screen of source/overlay pairs rated 23 images visually
plausible, 13 uncertain, and 0 obvious mismatches. These are provisional,
subjective ratings, not pose-accuracy labels or ground truth; see
`benchmarks/results/AI_VISUAL_AUDIT.txt`.
