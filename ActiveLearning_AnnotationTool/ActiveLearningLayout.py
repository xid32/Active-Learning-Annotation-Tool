

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

from Widgets.SakuraPlotWidget import SakuraPlotWidget
from Widgets.VideoWidget import VideoWidget

from Sensors.VideoSensor import VideoSensor

import cv2

import os, time


class ActiveLearningLayout(QVBoxLayout):

    current_video = None

    frameChanged = pyqtSignal(int)

    def __init__(self, settings, time_axis):
        super().__init__()

        data_paths = settings['data_paths']
        self.sync = settings['sync']
        self.with_video = True
        # self.fps = settings['fps']

        h_layout = QHBoxLayout()

        if data_paths[0] == '':
            self.with_video = False

        self.current_video = settings['current_video']
        self.ts = SakuraPlotWidget(settings['current_csv'], time_axis)
        self.setup_metadata()

        print("settings['current_video']:", settings['current_video'])
        self.video = VideoWidget(self.current_video, self.fps)

        #self.current_video = data_paths[0]

        self.addWidget(self.video)

        self.addWidget(self.ts)

        self.addLayout(h_layout)

        self.frameChanged.connect(self.ts.on_frame_change)

    def lazy_init(self, settings):
        pass


    def reload_csv_by_video(self, video_file, base_dir, p, time_t):
        self.ts.reload_csv_by_video(video_file, base_dir, p, time_t)

    def reload_csv(self, base_dir, p, time_t):
        self.ts.reload_csv(p, base_dir, time_t)

    def setup_metadata(self):
        file_name = os.path.split(self.current_video)[1]
        ts = int(
            time.mktime(time.strptime(os.path.split(file_name)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI'))
        )
        capture = cv2.VideoCapture(self.current_video)
        total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = capture.get(cv2.CAP_PROP_FPS)

        self.ts.setup(ts, self.fps, total_frame)

    def video_reentrant(self, path):
        self.current_video = path
        del self.video.sensor
        self.video.sensor = VideoSensor(None)
        self.video.sensor.load_data(path)
        self.update_components(0)
        self.setup_metadata()

    def update_components(self, frame):
        self.video.update(frame)
        #self.ts.update_by_frame(frame)
        self.frameChanged.emit(frame)


    def get_len(self):
        return self.video.get_len()

    def get_time(self):
        return self.video.get_time()
