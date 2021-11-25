

import re
from ui_Timeline import Ui_Timeline

from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget, QMainWindow, QApplication, QSlider, QLabel, QAbstractItemView
from PyQt5.QtWidgets import QHeaderView

import csv, os, time

import cv2


class Timeline(Ui_Timeline):

    sense_view = None
    video_widget = None
    csv_manager = None
    video_file_path = None

    data_set = None

    start_time = 0

    total_frame = 0
    fps = 0
    total_time = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.resize(600, 750)

    def attach_to(self, window):
        rect = window.frameGeometry()
        tr = rect.topRight()
        self.move(QPoint(tr.x() + 1, tr.y()))

    def on_cell_double_clicked(self, row, column):

        p, time_t_start, time_t_end = self.csv_manager.get_navigate_to_arguments(row)
        self.sense_view.showNormal()
        self.sense_view.navigate_to(p, time_t_start, time_t_end, row)

    def reload(self, csv_manager):
        self.csv_manager = csv_manager
        self.csv_manager.setup_table(self.u_timeline_table)

    def setup(self, sense_view, video_widget, csv_manager, video_file_path):
        print('Timeline', (self, sense_view, video_widget, csv_manager, video_file_path))
        self.sense_view = sense_view
        self.video_widget = video_widget
        self.csv_manager = csv_manager
        self.video_file_path = video_file_path

        if not os.path.isfile(self.video_file_path): return False


        self.u_timeline_table.setColumnCount(4)

        self.u_timeline_table.setVerticalHeaderLabels(
            ('ID', 'start time', 'end time', 'labeling status')
        )
        self.u_timeline_table.horizontalHeader().resizeSection(0, 256)
        self.u_timeline_table.horizontalHeader().resizeSection(1, 256)
        self.u_timeline_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.u_timeline_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.u_timeline_table.setSelectionMode( QAbstractItemView.SingleSelection)
        self.u_timeline_table.cellDoubleClicked.connect(self.on_cell_double_clicked)

        self.u_timeline_table.verticalHeader().setVisible(False)

        self.u_timeline_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.u_timeline_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.u_timeline_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.u_timeline_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)


        self.csv_manager.setup_table(self.u_timeline_table)


        capture = cv2.VideoCapture(self.video_file_path)
        self.total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = capture.get(cv2.CAP_PROP_FPS)

        return True

    def reject(self):
        exit(0)
