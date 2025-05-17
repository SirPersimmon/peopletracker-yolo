import os
import pathlib

import cv2
import numpy as np

import api.deepsort
import api.yolo

import deepsort.detection
import deepsort.nn_matching
import deepsort.tracker

if "MODEL_PATH" not in os.environ:
    raise Exception("Environment variable 'MODEL_PATH' is not set")

MAX_COSINE_DISTANCE = 0.5
MAX_AGE = 100


class EnhancedTracker(deepsort.tracker.Tracker):
    def __init__(
        self, metric, max_iou_distance: float = 0.7, max_age: int = 30, n_init: int = 3
    ) -> None:
        super().__init__(metric, max_iou_distance, max_age, n_init)
        self.stats = {}

    def update(self, detections, frame_number: int) -> None:
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self._update_stats(frame_number)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _stat(self, start_frame: int) -> dict[str, int]:
        return {"start_frame": start_frame, "duration": 1}

    def _update_stats(self, frame_number: int) -> None:
        for track in self.tracks:
            if track.track_id not in self.stats:
                self.stats[track.track_id] = self._stat(frame_number)
            else:
                stat = self.stats[track.track_id]
                stat["duration"] = track.age


class EnchancedDeepSORTTracker(api.deepsort.DeepSORTTracker):
    def __init__(
        self,
        reid_model: str,
        cosine_thresh: float,
        max_track_age: int,
        nn_budget: int = None,
    ) -> None:
        super().__init__(reid_model, cosine_thresh, max_track_age, nn_budget)
        self.tracker = EnhancedTracker(
            deepsort.nn_matching.NearestNeighborDistanceMetric(
                "cosine", cosine_thresh, nn_budget
            ),
            max_age=max_track_age,
        )

    def track(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        frame_number: int,
    ) -> None:
        """
        Accepts an image and its YOLO detections, uses these detections and existing tracks to get a
        final set of bounding boxes, which are then drawn onto the input image
        """
        feats = self.encoder(frame, bboxes)
        dets = [
            deepsort.detection.Detection(*args) for args in zip(bboxes, scores, feats)
        ]

        # refine the detections
        self.tracker.predict()
        self.tracker.update(dets, frame_number)

        # render the final tracked bounding boxes on the input frame
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(np.int32)
            color = self.colors[track.track_id % 20]
            # draw detection bounding box
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            # draw text box for printing ID
            cv2.rectangle(
                frame,
                tuple(bbox[:2]),
                (bbox[0] + (4 + len(str(track.track_id))) * 8, bbox[1] + 20),
                color,
                -1,
            )
            # print ID in the text box
            cv2.putText(
                frame,
                f"ID: {track.track_id}",
                (bbox[0] + 4, bbox[1] + 13),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (0, 0, 0),
                lineType=cv2.LINE_AA,
            )


class Input:
    def __init__(self, file_path: str = "") -> None:
        self.detector = api.yolo.YOLOPersonDetector()
        self.detector.load(
            (pathlib.Path(os.environ["MODEL_PATH"]) / "yolov7x.pt").resolve()
        )

        self.tracker = EnchancedDeepSORTTracker(
            (pathlib.Path(os.environ["MODEL_PATH"]) / "ReID.pb").resolve(),
            MAX_COSINE_DISTANCE,
            MAX_AGE,
        )

        self.capture = cv2.VideoCapture(file_path if file_path else 0)
        if self.capture.isOpened():
            self.frameSize = (
                int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            self.frameRate = int(self.capture.get(cv2.CAP_PROP_FPS))
        else:
            self.frameSize = (0, 0)
            self.frameRate = 0

        self.frame_number = 0

    def run(self) -> bool:
        result, self.currentFrame = self.capture.read()

        if not result:
            return result

        self.frame_number += 1

        self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2RGB)

        # process YOLO detections
        detections = self.detector.detect(self.currentFrame)
        try:
            bboxes, scores, _ = np.hsplit(detections, [4, 5])
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
            n_objects = detections.shape[0]
        except ValueError:
            bboxes = np.empty(0)
            scores = np.empty(0)
            n_objects = 0

        # track targets by refining with DeepSORT
        self.tracker.track(
            self.currentFrame, bboxes, scores.flatten(), self.frame_number
        )

        self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_RGB2BGR)

        return result

    def get_stats(self):
        return self.tracker.tracker.stats


class Analyzer:
    def __init__(self, input_video_path="", output_video_path="", fourcc="mp4v"):
        self.input = Input(input_video_path)
        self.output_video_path = output_video_path
        self.fourcc = fourcc

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        output_video = cv2.VideoWriter(
            self.output_video_path,
            fourcc=fourcc,
            fps=self.input.frameRate,
            frameSize=self.input.frameSize,
        )
        while True:
            result = self.input.run()
            if not result:
                break

            output_video.write(self.input.currentFrame)

        output_video.release()

    def get_stats(self):
        return self.input.get_stats()
