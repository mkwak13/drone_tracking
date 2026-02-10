import numpy as np
from filterpy.kalman import KalmanFilter

# calculate IoU between two bboxes
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6
    )

# tracker
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)
        self.kf.P *= 10.
        self.kf.R *= 1.
        self.kf.Q *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox.reshape((4, 1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

class Sort:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.trackers = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, detections):
        results = []

        predicted_boxes = [trk.predict() for trk in self.trackers]
        matched = set()

        for det in detections:
            best_iou = self.iou_threshold
            best_trk = None

            for trk, pred in zip(self.trackers, predicted_boxes):
                if trk in matched:
                    continue
                score = iou(det[:4], pred)
                if score > best_iou:
                    best_iou = score
                    best_trk = trk

            if best_trk is not None:
                best_trk.update(det[:4])
                matched.add(best_trk)
                results.append([*det[:4], best_trk.id])
            else:
                trk = KalmanBoxTracker(det[:4])
                self.trackers.append(trk)
                results.append([*det[:4], trk.id])

        # remove outdated tracker
        self.trackers = [
            trk for trk in self.trackers
            if trk.time_since_update <= self.max_age
        ]

        return np.array(results)
