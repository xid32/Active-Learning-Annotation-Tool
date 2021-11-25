

import numpy as np
import cv2

from Util import VideoMetadata

class VideoSensor:

    def __init__(self, path):
        self.total_frames = 0
        self.fps = 0
        self.time = None
        if path != None and len(path) > 0:
            self.data = self.load_data(path)

    def load_data(self, video_path):
        self.video_path = video_path
        self.data = cv2.VideoCapture(self.video_path)

        self.set_fps(VideoMetadata.get_meta(video_path).fps)

    def set_fps(self, fps):
        self.fps = fps
        self.total_frames = VideoMetadata.get_meta(self.video_path).total_frame
        self.duration =  self.total_frames / self.fps
        self.time = np.arange(self.total_frames)

    def get_image(self, index):
        self.data.set(cv2.CAP_PROP_POS_MSEC, index * 1000 / self.fps)
        _, frame = self.data.read()
        return frame

    def get_time(self):
        return self.time

