import cv2 as cv

from .calc import CALC
from .camera import CAMERA
from .draw import DRAW
from .model import MODEL
from .options import OPTIONS


class MODULE:
    mode = 0
    number = -1
    point_history = OPTIONS.point_history
    hand_landmarks = 0

    def __init__(self, calc_module=None, classifier_model=None, dr=None, DEBUG=False):
        self.cl = calc_module if calc_module is not None else CALC()
        self.classifier_model = classifier_model if classifier_model is not None else MODEL()
        self.draw_module = dr if dr is not None else DRAW()
        self.DEBUG = DEBUG

    def quit(self):
        CAMERA.cap.release()
        cv.destroyAllWindows()

    def select_mode(self, key, mode):
        if 48 <= self.key <= 57:  # 0 ~ 9
            self.number = self.key - 48
        if self.key == 110:  # n
            self.mode = 0
        if self.key == 107:  # k
            self.mode = 1
        if self.key == 104:  # h
            self.mode = 2
        return self.number, self.mode

    def keyFunc(self, debug_image):
        info_text = {}
        results = self.classifier_model.detect(debug_image)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                brect = self.cl.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.cl.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.cl.pre_process_landmark(landmark_list)

                hand_sign_id = self.classifier_model.keypoint_classifier(pre_processed_landmark_list)
                hand_sign = self.classifier_model.keypoint_classifier_labels[hand_sign_id]
                hand = handedness.classification[0].label

                if self.DEBUG:
                    debug_image = self.draw_module.bounding_rect(True, debug_image, brect)
                    debug_image = self.draw_module.landmarks(debug_image, landmark_list)
                    debug_image = self.draw_module.info_text(debug_image, brect, hand, hand_sign)

                info_text[hand] = hand_sign

            return hand_landmarks, debug_image, info_text
        else:
            return [[0, 0]], debug_image, info_text
