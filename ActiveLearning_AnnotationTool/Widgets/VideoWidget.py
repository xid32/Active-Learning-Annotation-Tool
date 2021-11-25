

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap

from Sensors.VideoSensor import VideoSensor

import numpy as np
import qimage2ndarray


class VideoWidget(QLabel):

    def __init__(self, data_path, fps):
        super().__init__()
        self.sensor = VideoSensor(None)
        self.sensor_data = self.sensor.load_data(data_path)
        self.setMinimumSize(640, 360)
        black_image = np.zeros([640, 360, 3], dtype=np.uint8)
        black_image.fill(0)
        self.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(black_image)))
        
    def get_len(self):
        return self.sensor.total_frames

    def update(self, frame):
        image = self.sensor.get_image(frame)
        pixmap =  QPixmap.fromImage(qimage2ndarray.array2qimage(image))
        self.setPixmap(pixmap)

    def get_time(self):
        return self.sensor.get_time()

