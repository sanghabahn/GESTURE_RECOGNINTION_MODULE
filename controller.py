import time

import cv2 as cv
from pynput.keyboard import Key, Controller

from library.calc import CALC
from library.camera import CAMERA
from library.draw import DRAW
from library.model import MODEL
from library.module import MODULE
from library.options import OPTIONS

input_dictionary = {
    "Left:Go": Key.right,
    "Left:Back": Key.left,
    "Right:Jump": Key.space,

    "Left:Go Right:Jump": Key.right,
    "Left:Back Right:Jump": Key.left,
    "Left:STOP Right:Jump": Key.left,
    "Left:Go Right:STOP": Key.right,
    "Left:Back Right:STOP": Key.left,
    "Left:STOP Right:STOP": Key.left,
}


class inputController:
    def __init__(self, DEBUG=True):
        self.DEBUG = DEBUG
        self.options = OPTIONS()
        self.cam = CAMERA()
        self.dr = DRAW()
        self.cl = CALC()
        self.cam.prep()
        self.classifier = MODEL()
        self.classifier.read_labels()
        self.md = MODULE(self.cl, self.classifier, self.dr)
        self.keyboard = Controller()

    def _press_jump(self):
        self.keyboard.press(Key.space)
        time.sleep(1/60)
        self.keyboard.release(Key.space)

    def _press_move(self, dir):
        self.keyboard.press(Key.right if dir == "R" else Key.left)

    def _stop_move(self, dir):
        self.keyboard.release(Key.right if dir == "R" else Key.left)

    def _control(self, info_text):
        if "Left" in info_text:
            if info_text["Left"] == "Go":
                self._press_move("R")
                # print("R",end="")
            elif info_text["Left"] == "Back":
                self._press_move("L")
                # print("L",end="")
            elif info_text["Left"] == "STOP":
                self._stop_move("L")
                self._stop_move("R")
                # print("Stop_move",end="")
        if "Right" in info_text:
            if info_text["Right"] == "Jump":
                self._press_jump()

    def run(self):
        debug_img, display = self.cam.capture()

        if not self.DEBUG:
            self.cam.display(debug_img)
            dtc_rst = self.cam.detect(display)
            hand_landmarks, debug_image, info_text = self.md.keyFunc(dtc_rst, debug_img)
            self._control(info_text)

        if self.DEBUG:
            dtc_rst = self.cam.detect(display)
            hand_landmarks, debug_image, info_text = self.md.keyFunc(dtc_rst, debug_img)
            self.cam.display(debug_image)
            self._control(info_text)

    def quit(self):
        self.md.quit()


if __name__ == "__main__":

    running_module = inputController(True)  # True : landmarks on  /  False : landmarks off
    while True:
        running_module.run()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            running_module.quit()
            break
