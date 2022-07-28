import cv2
import sys
from random import randint


class SingleObjectTracking:
    def __init__(self, tracker, video):
        self.tracker_type = tracker.upper()
        self.video = video

    def get_tracker(self):
        if self.tracker_type == "BOOSTING":
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == "MIL":
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "TLD":
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == "MEDIANFLOW":
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        else:
            print("tracker not found!")
            sys.exit()

    def start_tracking(self):
        tracker = self.get_tracker()
        video = self.get_video()
        ok, first_frame = video.read()
        if not ok:
            print("Error while loading the frame!")
            sys.exit()
        first_bbox = self.select_initial_bbox(first_frame)
        bbox_color = self.get_bbox_color()
        tracker.init(first_frame, first_bbox)
        while True:
            ok, frame = video.read()
            if not ok:
                break
            ok, bbox = tracker.update(frame)
            if ok:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2, 1)
            else:
                cv2.putText(
                    frame,
                    "Tracking faliure!",
                    (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    @staticmethod
    def select_initial_bbox(frame):
        return cv2.selectROI(frame)

    def get_video(self):
        video = cv2.VideoCapture(self.video)
        if not video.isOpened():
            print("Error while loading the video!")
            sys.exit()
        return video

    @staticmethod
    def get_bbox_color():
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        return color
