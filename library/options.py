import dataclasses
from collections import deque

@dataclasses.dataclass
class OPTIONS:
    device = 0
    width = 960
    height = 540
    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    history_length = 16
    point_history = deque(maxlen=history_length)

    def __init__(self):
        pass