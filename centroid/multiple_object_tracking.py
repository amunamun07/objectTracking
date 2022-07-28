import cv2
import sys
from random import randint


class MultipleObjectTracking:
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

