import copy

import cv2 as cv
import numpy as np


class CALC:
    def __init__(self):
        pass

    def key_point_enumerate(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        ret_landmark_list = []

        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            ret_landmark_list.append([landmark_x, landmark_y])

        return ret_landmark_list

    def calc_bounding_rect(self, image, landmarks):
        ret_landmark_list = self.key_point_enumerate(image, landmarks)
        ret_landmark_np = np.array(ret_landmark_list)

        x, y, w, h = cv.boundingRect(ret_landmark_np)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        ret_landmark_list = self.key_point_enumerate(image, landmarks)

        return ret_landmark_list

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        temp_landmark_list = np.array(temp_landmark_list, dtype=np.float64)
        temp_landmark_list -= temp_landmark_list[0]

        max_value = np.max(np.abs(temp_landmark_list))
        temp_landmark_list /= max_value
        temp_landmark_list = temp_landmark_list.flatten()

        return temp_landmark_list
