import cv2 as cv


class DRAW:
    fingers_connected_indices = [(2, 3), (3, 4),  # thumb
                                 (5, 6), (6, 7), (7, 8),  # index finger
                                 (9, 10), (10, 11), (11, 12),  # middle finger
                                 (13, 14), (14, 15), (15, 16),  # ring finger
                                 (17, 18), (18, 19), (19, 20)  # pinky
                                 ]
    palm_connected_indices = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]

    def __init__(self):
        pass

    def landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            for index1, index2 in self.fingers_connected_indices:
                cv.line(image, tuple(landmark_point[index1]), tuple(landmark_point[index2]), (0, 0, 0), 6)
                cv.line(image, tuple(landmark_point[index1]), tuple(landmark_point[index2]), (255, 255, 255), 2)

            for index1, index2 in self.palm_connected_indices:
                cv.line(image, tuple(landmark_point[index1]), tuple(landmark_point[index2]), (0, 0, 0), 6)
                cv.line(image, tuple(landmark_point[index1]), tuple(landmark_point[index2]), (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index not in [4, 8, 12, 16, 20]:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            else:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

        return image

    def info_text(self, image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text

        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv.LINE_AA)

        return image
