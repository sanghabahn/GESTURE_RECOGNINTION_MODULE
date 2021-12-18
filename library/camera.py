import copy

import cv2 as cv

from .options import OPTIONS


class CAMERA:
    cap_device = OPTIONS.device
    cap_width = OPTIONS.width
    cap_height = OPTIONS.height
    cap = cv.VideoCapture(cap_device)
    use_brect = True

    def __init__(self):
        self.prep()
        # pass

    def prep(self):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

    def capture(self):
        ret, image = self.cap.read()
        display = cv.flip(image, 1)  # Mirror display
        debug_img = copy.deepcopy(display)
        return debug_img, display

    def display(self, debug_image):
        cv.imshow('Hand Gesture Recognition', debug_image)
