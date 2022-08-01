import os.path

import cv2
import sys
from random import randint


class MultipleObjectTracking:
    def __init__(self, tracker, video_path):
        self.tracker_type = tracker.upper()
        self.video_path = video_path

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

    @staticmethod
    def select_initial_bbox(frame):
        return cv2.selectROI("MultiTracker", frame)

    def get_video(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print("Error while loading the video!")
            sys.exit()
        return video

    @staticmethod
    def get_bbox_color():
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        return color

    def start_tracking(self):
        tracker = self.get_tracker()
        video = self.get_video()
        bboxes = []
        bboxes_colors = []
        ok, first_frame = video.read()
        if not ok:
            print("Error while loading the frame!")
            sys.exit()
        while True:
            bbox = self.select_initial_bbox(first_frame)
            bboxes.append(bbox)
            bboxes_colors.append(self.get_bbox_color())
            print("Press Q to quit and start the tracking")
            print("Press Any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if k == 113:
                break
        multi_tracker = cv2.legacy.MultiTracker_create()
        for box in bboxes:
            multi_tracker.add(tracker, first_frame, box)
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            else:
                ok, boxes = multi_tracker.update(frame)
                for i, newbox in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in newbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bboxes_colors[i], 2)

                cv2.imshow("MultiTracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
