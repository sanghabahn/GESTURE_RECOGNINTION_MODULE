import csv

import cv2 as cv
import mediapipe as mp

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from .options import OPTIONS


class MODEL:

    def __init__(self):
        self.keypoint_classifier = KeyPointClassifier()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=OPTIONS.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=OPTIONS.min_detection_confidence,
            min_tracking_confidence=OPTIONS.min_tracking_confidence,
        )
        self.read_labels()

    def read_labels(self):
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            data = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in data]

    def detect(self, img):
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        return results
