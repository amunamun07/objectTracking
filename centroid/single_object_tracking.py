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
        elif self.tracker_type == "DASIAMRPN":
            return cv2.legacy.TrackerDaSiamRPN_create()
        elif self.tracker_type == "GOTURN":
            return cv2.legacy.TrackerGOTURN_create()
        else:
            print("tracker not found!")
            sys.exit()

    def start_tracking(self):
        tracker = self.get_tracker()
        video = self.get_video()
        while True:
            ok, initial_frame = video.read()
            if not ok:
                break
            frm = cv2.resize(initial_frame, (960, 540))
            cv2.imshow("Tracking", frm)
            k = cv2.waitKey(0) & 0xFF
            if k == 113:
                cv2.destroyAllWindows()
                break
        first_bbox = self.select_initial_bbox(frm)
        bbox_color = self.get_bbox_color()
        tracker.init(frm, first_bbox)
        while True:
            ok, frame = video.read()
            frm = cv2.resize(frame, (960, 540))
            if not ok:
                break
            ok, bbox = tracker.update(frm)
            if ok:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frm, (x, y), (x + w, y + h), bbox_color, 2, 1)
            else:
                cv2.putText(
                    frm,
                    "Tracking faliure!",
                    (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Tracking", frm)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    @staticmethod
    def select_initial_bbox(frame):
        return cv2.selectROI("Single Tracker", frame)

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
