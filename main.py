import sys
import time
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gpu_renderer import GpuRenderer, self_check as gpu_self_check

MODEL = Path(__file__).with_name("hand_landmarker.task")
CAM_INDEX = 0
DET_W = 640
NUM_HANDS = 2
MIN_SPAN = 40
MAX_MISSED = 6
MAX_PREDICTION = 0.05
SNAP_RATIO = 0.5
SNAP_COOLDOWN = 0.6

WRIST, THUMB, INDEX, MIDDLE, PINKY, MIDDLE_MCP = 0, 4, 8, 12, 20, 9
PANEL_ANCHORS = (INDEX, MIDDLE, PINKY, THUMB)


class OneEuroHandFilter:
    def __init__(self, min_cutoff=1.5, beta=2.0, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.raw = None
        self.value = None
        self.velocity = None
        self.timestamp = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    def update(self, points, timestamp):
        points = np.asarray(points, np.float32)
        if self.value is None:
            self.raw = points.copy()
            self.value = points.copy()
            self.velocity = np.zeros_like(points)
            self.timestamp = timestamp
            return self.value.copy()

        dt = np.clip(timestamp - self.timestamp, 1e-3, 0.1)
        raw_velocity = (points - self.raw) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self.velocity += derivative_alpha * (raw_velocity - self.velocity)

        hand_scale = max(
            float(np.linalg.norm(points[WRIST] - points[MIDDLE_MCP])), 1.0
        )
        normalized_speed = np.linalg.norm(self.velocity, axis=1, keepdims=True) / hand_scale
        cutoff = self.min_cutoff + self.beta * normalized_speed
        alpha = self._alpha(cutoff, dt)
        self.value += alpha * (points - self.value)
        self.raw = points.copy()
        self.timestamp = timestamp
        return self.value.copy()


class HandTrack:
    def __init__(self, points, timestamp):
        self.filter = OneEuroHandFilter()
        self.points = self.filter.update(points, timestamp)
        self.missed = 0

    def update(self, points, timestamp):
        self.points = self.filter.update(points, timestamp)
        self.missed = 0

    def projected(self, timestamp):
        if self.missed or self.filter.timestamp is None:
            return self.points.copy()
        horizon = np.clip(timestamp - self.filter.timestamp, 0.0, MAX_PREDICTION)
        displacement = self.filter.velocity * horizon
        hand_scale = max(
            float(
                np.linalg.norm(
                    self.points[WRIST] - self.points[MIDDLE_MCP]
                )
            ),
            1.0,
        )
        distance = np.linalg.norm(displacement, axis=1, keepdims=True)
        displacement *= np.minimum(1.0, 0.2 * hand_scale / (distance + 1e-6))
        return self.points + displacement


def update_hand_tracks(tracks, detections, timestamp):
    for track in tracks:
        track.missed += 1

    matches = []
    if len(tracks) == 2 and len(detections) == 2:
        distances = np.array(
            [
                [
                    np.linalg.norm(track.points[WRIST] - points[WRIST])
                    for points in detections
                ]
                for track in tracks
            ]
        )
        direct = distances[0, 0] + distances[1, 1]
        swapped = distances[0, 1] + distances[1, 0]
        order = (0, 1) if direct <= swapped else (1, 0)
        matches = [(0, order[0]), (1, order[1])]
    elif tracks and detections:
        candidates = sorted(
            (
                np.linalg.norm(track.points[WRIST] - points[WRIST]),
                track_index,
                detection_index,
            )
            for track_index, track in enumerate(tracks)
            for detection_index, points in enumerate(detections)
        )
        used_tracks = set()
        used_detections = set()
        for _distance, track_index, detection_index in candidates:
            if track_index not in used_tracks and detection_index not in used_detections:
                matches.append((track_index, detection_index))
                used_tracks.add(track_index)
                used_detections.add(detection_index)

    matched_detections = set()
    for track_index, detection_index in matches:
        tracks[track_index].update(detections[detection_index], timestamp)
        matched_detections.add(detection_index)

    tracks[:] = [track for track in tracks if track.missed <= MAX_MISSED]
    for detection_index, points in enumerate(detections):
        if detection_index not in matched_detections and len(tracks) < NUM_HANDS:
            tracks.append(HandTrack(points, timestamp))

    return [track.points for track in tracks]


def projected_hands(tracks, timestamp):
    return [track.projected(timestamp) for track in tracks]


def to_px(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], np.float32)


def mask_quads_from_hands(hands):
    hands = sorted(hands, key=lambda p: p[WRIST, 0])
    left, right = hands
    boundaries = [(left[anchor], right[anchor]) for anchor in PANEL_ANCHORS]
    return [
        np.float32([left_a, right_a, right_b, left_b])
        for (left_a, right_a), (left_b, right_b) in zip(boundaries, boundaries[1:])
    ]


def is_pinched(p):
    hand = np.linalg.norm(p[WRIST] - p[MIDDLE_MCP]) + 1e-6
    return np.linalg.norm(p[THUMB] - p[MIDDLE]) < SNAP_RATIO * hand


def self_check():
    left = np.zeros((21, 2), np.float32)
    right = np.zeros((21, 2), np.float32)
    left[[INDEX, MIDDLE, PINKY]] = [(20, 18), (25, 40), (30, 65)]
    right[[INDEX, MIDDLE, PINKY]] = [(130, 23), (135, 45), (140, 70)]
    left[THUMB] = (35, 100)
    right[THUMB] = (125, 105)
    left[WRIST] = (15, 80)
    right[WRIST] = (145, 80)

    quads = mask_quads_from_hands([right, left])
    assert len(quads) == 3
    assert np.allclose(quads[0], [[20, 18], [130, 23], [135, 45], [25, 40]])
    assert np.allclose(quads[2], [[30, 65], [140, 70], [125, 105], [35, 100]])
    assert np.allclose(quads[0][2:], quads[1][1::-1])
    assert np.allclose(quads[1][2:], quads[2][1::-1])

    hand = np.zeros((21, 2), np.float32)
    hand[MIDDLE_MCP] = (0, 50)
    hand[WRIST] = (0, 100)
    hand[INDEX] = (40, 30)
    filter_ = OneEuroHandFilter()
    raw = []
    filtered = []
    for frame in range(30):
        noisy = hand.copy()
        noisy[:, 0] += 3.0 if frame % 2 else -3.0
        raw.append(noisy[INDEX, 0])
        filtered.append(filter_.update(noisy, frame / 30.0)[INDEX, 0])
    assert np.std(filtered[5:]) < np.std(raw[5:]) * 0.5

    left_track = left.copy()
    right_track = right.copy()
    tracks = []
    update_hand_tracks(tracks, [left_track, right_track], 0.0)
    update_hand_tracks(tracks, [right_track + (2, 0), left_track + (2, 0)], 1 / 30)
    assert tracks[0].points[WRIST, 0] < tracks[1].points[WRIST, 0]
    print("self-check ok")


def main():
    latest_result = [(-1, None)]

    def save_result(result, _image, timestamp_ms):
        latest_result[0] = (timestamp_ms, result)

    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=save_result,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    mode = 0
    tracks = []
    pinched_prev = False
    last_snap = 0.0
    start = time.perf_counter()
    last_ts = 0
    processed_ts = -1
    hands = []
    renderer = None

    try:
        renderer = GpuRenderer(
            camera_index=CAM_INDEX,
            detect_width=DET_W,
        )
        width, height = renderer.width, renderer.height
        print(renderer.camera_info())
        renderer.start()
        running = True
        while running:
            small = renderer.pull_detection()
            if small is None:
                continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
            ts = max(last_ts + 1, int((time.perf_counter() - start) * 1000))
            last_ts = ts
            landmarker.detect_async(mp_img, ts)

            result_ts, result = latest_result[0]
            if result_ts != processed_ts:
                processed_ts = result_ts
                detections = []
                if result:
                    detections = [
                        to_px(landmarks, width, height)
                        for landmarks in result.hand_landmarks
                    ]
                update_hand_tracks(tracks, detections, result_ts / 1000.0)

            now = time.perf_counter()
            hands = projected_hands(tracks, now - start)
            pinched = any(is_pinched(hand) for hand in hands)
            if pinched and not pinched_prev and now - last_snap > SNAP_COOLDOWN:
                mode = (mode + 1) % 3
                last_snap = now
            pinched_prev = pinched

            quads = None
            if len(hands) >= 2:
                candidates = mask_quads_from_hands(hands)
                left, right = sorted(hands, key=lambda p: p[WRIST, 0])
                spans = [
                    np.linalg.norm(left[anchor] - right[anchor])
                    for anchor in PANEL_ANCHORS
                ]
                if np.median(spans) > MIN_SPAN:
                    quads = candidates

            renderer.update(quads, mode=mode, phase=now - start)
            key = renderer.pop_key()
            while key is not None:
                if key in ("q", "escape"):
                    running = False
                elif key in ("n", "space", " "):
                    mode = (mode + 1) % 3
                elif key == "p":
                    mode = (mode - 1) % 3
                key = renderer.pop_key()
    except KeyboardInterrupt:
        pass
    finally:
        if renderer is not None:
            renderer.stop()
        landmarker.close()


if __name__ == "__main__":
    if "--gpu-check" in sys.argv:
        gpu_self_check()
    elif "--self-check" in sys.argv:
        self_check()
    else:
        main()
