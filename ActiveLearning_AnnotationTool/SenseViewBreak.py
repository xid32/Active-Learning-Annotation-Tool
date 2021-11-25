

from PyQt5.QtGui import QFont
from OutputHook import *

from Verify import *

from cv2 import data
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yaml
from PyQt5.QtCore import QEvent, QPoint, QTimer, Qt
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget, QMainWindow, QApplication, QSlider, QLabel

from PyQt5.QtWidgets import QMessageBox

import pyqtgraph as pg

import argparse

from Timeline import Timeline

import csv
import signal
import sys
import traceback

from TimeAxis import TimeAxis, TimeAxisForSakuraPlot

import json

import Util
from Util import VideoMetadata

from ActivelearningOutputManager import ActivelearningOutputManager

from ActiveLearningLayout import ActiveLearningLayout

from ActiveLearning import ActiveLearning

# from Widgets.FrameBarWidget import FrameBarWidget

from Config import FILE_FORMAT

from Util import TimeT

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGINT, signal.SIG_DFL)

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, help='path to input (image folder or ' + FILE_FORMAT +  ')')


PATH_MAPPING_TABLE = {
    'P101':'P203-2',
    'P102':'P205',
    'P103':'P206',
    'P104':'P207',
    'P105':'P209',
    'P106':'P210',
    'P107':'P211',
    'P108':'P212',
    'P109':'P217',
    'P110':'P218'
}


class SenseView(QMainWindow):
    frame = 0
    max_frames = 2147483647
    timeline = None

    current_video = None
    current_csv = None
    current_video_meta = None
    current_p = None
    mode = None

    flag_sliderPressed = False

    activelearning_output = None

    @property
    def current_frame(self):
        return self.frame

    @current_frame.setter
    def current_frame(self, current_frame):
        self.frame = current_frame

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            if self.timeline.isVisible(): self.timeline.showNormal()
            if self.active_learning_window.isVisible(): self.active_learning_window.showNormal()

        super().changeEvent(event)

    def __init__(self, args, timeline):
        super().__init__()

        self.pending_task = []

        try:
            os.mkdir('export')
        except:
            pass

        self.move(QPoint(301, 153))

        self.timeline = timeline

        self.window = QWidget()
        self.setCentralWidget(self.window)
        self.main_layout = QVBoxLayout()
        self.window.setLayout(self.main_layout)

        ########  adding the player controls, slider and timer ########
        # Play and pause button
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setText('Play')
        self.playing = False
        self.play_pause_btn.clicked.connect(self.play_pause)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.sliderReleased.connect(self.apply_slider_change)
        self.slider.sliderPressed.connect(self.sliderPressed)

        # adding slider and player to a layout
        self.player_controls = QHBoxLayout()
        self.player_controls.addWidget(self.play_pause_btn)
        self.player_controls.addWidget(self.slider)

        

        # setting up a timer to play the video automatically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.timer_update)

        ########   adding annotation related code ########

        self.labels_view = QHBoxLayout()
        self.labeling = False

        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        view.setMaximumHeight(100)
        self.labels_view.addWidget(view)

        self.time_asix = TimeAxis(self, orientation='bottom')
        self.time_asix_for_plot = TimeAxisForSakuraPlot(self, orientation='bottom')

        self.w1 = view.addPlot(axisItems={'bottom': self.time_asix})
        self.w1.setMouseEnabled(y=False)
        self.w1.setXRange(0, 50, padding=None)
        self.w1.setYRange(0, 3, padding=None)
        #ax_b = self.w1.getAxis('bottom')  # This is the trick




        #FBW

        #self.frame_buffer_widget = FrameBarWidget(self)
        #self.main_layout.addWidget(self.frame_buffer_widget)

        self.s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        self.s1.sigClicked.connect(self.clicked)
        self.w1.addItem(self.s1)

        self.time_cursor = pg.InfiniteLine(movable=True, angle=90)
        self.time_cursor.sigDragged.connect(self.dragged)
        self.w1.addItem(self.time_cursor)

        ## Make all plots clickable
        self.lastClicked = []
        self.labels = []
        self.loaded_labels = None

        import_export_view = QVBoxLayout()

        self.labeling_l = QLabel()
        self.labeling_l.setText('')
        import_export_view.addWidget(self.labeling_l)


        self.timestamp = QLabel()
        self.timestamp.setText('Timestamp:')
        # import_export_view.addWidget(self.timestamp)

        self.current_index = QLabel()
        self.current_index.setText('Frame:0')
        # import_export_view.addWidget(self.current_index)

        # Annotate as chewing sequence
        # Annotate as non-chewing sequence

        as_chewing_btn = QPushButton()
        as_chewing_btn.setText('Annotate as chewing sequence    ')
        as_chewing_btn.clicked.connect(self.import_as_chewing)
        import_export_view.addWidget(as_chewing_btn)

        as_non_chewing_btn = QPushButton()
        as_non_chewing_btn.setText('Annotate as non-chewing sequence')
        as_non_chewing_btn.clicked.connect(self.import_as_non_chewing)
        import_export_view.addWidget(as_non_chewing_btn)

        export = QPushButton()
        export.setText('export')
        export.clicked.connect(self.export_labels)
        import_export_view.addWidget(export)

        import_from_config_btn = QPushButton()
        import_from_config_btn.setText('Import from yaml')
        import_from_config_btn.clicked.connect(self.import_from_config)
        import_export_view.addWidget(import_from_config_btn)

        
        self.labels_view.addLayout(import_export_view)

        data_paths = None
        remote_data_path = None
        self.label_path = None

        activelearning_output_file = None

        ######## User defined sensor layout #######
        with open(args.i) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            print('settings:', settings)
            if settings['remote_data_path'] == None and settings['data_paths'] == None:
                print('At least one of the `remote_data_path` and `data_paths` is needed.')
                exit(-1)
            
            self.al_backend_data = settings['al_backend_data']

            # data_paths = settings['data_paths']
            remote_data_path = settings['remote_data_path']

            self.label_path = settings['label_path']

        self.settings = settings

        target_video_file_path = None
        target_csv_file_path = None

        print('remote_data_path:', remote_data_path)

        if remote_data_path != None:
            self.mode = 'remote_data_path'
            
            if os.path.isdir(self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE')):
                print('`remote_data_path` found, switch to remote data path mode.')
                csv_file = open(self.settings['activelearning_output'][0], 'r')
                reader = csv.reader(csv_file)

                target_data = None
                p = None
                folder = None
                for line in reader:
                    folder = PATH_MAPPING_TABLE.get(line[0], 'NOT_FOUND')
                    if folder == 'NOT_FOUND': continue
                    p = line[0]
                    target_data = line
                    break

                self.current_p = p

                csv_file.close()
                print('target_data:', target_data)
                if target_data != None:
                    relocated_time = target_data[1]
                    relocated_datetime = relocated_time.split('.')[0]
                    relocated_timestamp = int(time.mktime(time.strptime(relocated_datetime, '%Y-%m-%d %H:%M:%S')))
                    thousandths = float('0.' + relocated_time.split('.')[1].split('-')[0])
                    selected_video_file = None

                    folder_base = self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE') + folder + os.sep + 'Camera'

                    print(Util.listdir(folder_base))
                    for file_index in range(len(Util.listdir(folder_base)) - 1):

                        file_start = Util.listdir(folder_base)[file_index]
                        file_end = Util.listdir(folder_base)[file_index + 1]

                        if not file_start.startswith('DVR___'): continue
                        if not file_end.startswith('DVR___'): continue

                        datetime_part_with_ext_name_start = file_start.split('___')[-1]
                        datetime_part_start = '.'.join(datetime_part_with_ext_name_start.split('.')[:-1])
                        start_time = int(time.mktime(time.strptime(datetime_part_start, '%Y-%m-%d_%H.%M.%S')))

                    
                        datetime_part_with_ext_name_end = file_end.split('___')[-1]
                        datetime_part_end = '.'.join(datetime_part_with_ext_name_end.split('.')[:-1])
                        end_time = int(time.mktime(time.strptime(datetime_part_end, '%Y-%m-%d_%H.%M.%S')))

                        if relocated_timestamp >= start_time and relocated_timestamp < end_time:
                            selected_video_file = Util.listdir(folder_base)[file_index]
                            break

                    print()
                    selected_video_file = 'DVR___2019-02-10_13.00.59.AVI'
                    print(folder_base, os.sep, selected_video_file)
                    target_video_file_path = folder_base + os.sep + selected_video_file



                    print('selected_video_file:', selected_video_file)

                    ymd = selected_video_file.split('_')[3]

                    y = int(ymd.split('-')[0])
                    m = int(ymd.split('-')[1])
                    d = int(ymd.split('-')[2])

                    csv_prefix = str(m).zfill(2) + '-' + str(d).zfill(2) + '-' + str(y)[-2:]

                    csv_prefix = os.path.splitext(selected_video_file)[0]
                    
                    csv_folder_base = self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE') + folder + os.sep + 'Necklace'

                    print('csv_folder_base:', csv_folder_base)
                    print('csv_folder_base file number:', len(Util.listdir(csv_folder_base)))

                    # relocated_timestamp_thousandths = (relocated_timestamp + thousandths) * 1000

                    if False:
                        relocated_timestamp_thousandths = Util.get_timestamp_from_video_file(target_video_file_path).ms

                        print('relocated_timestamp_thousandths:', relocated_timestamp_thousandths)
                        print('relocated_timestamp_thousandths:', time.localtime(int(relocated_timestamp_thousandths / 1000)))
                    
                    print('csv_prefix:', csv_prefix)

                    target_csv_file_name = None
                    target_csv_files = []

                    for file_name in Util.listdir(csv_folder_base):

                        if file_name.startswith(csv_prefix):
                            file_path = csv_folder_base + os.sep + file_name
                            target_csv_files.append(file_path)
                            continue

                        target_csv_file_path = csv_folder_base
                        self.settings['data_paths'] = [target_video_file_path, target_csv_files]
                        self.settings['current_csv'] = target_csv_file_path
                    
                    print(target_csv_files)
            

            self.current_video = target_video_file_path
            self.current_csv = target_csv_file_path

            

            print('self.current_video:', self.current_video)
        else:
            self.mode = 'data_paths'
            
            self.current_video = os.path.abspath(data_paths[0])
            self.current_csv = os.path.abspath(data_paths[1])


        if settings['activelearning_output'] == None or len(settings['activelearning_output']) == 0:
            pass
        else:
            self.activelearning_output = ActivelearningOutputManager(settings['activelearning_output'])


        original_p = self.activelearning_output.all_data.iat[0, 0]
        date_str = self.activelearning_output.all_data.iat[0, 1]

        p = PATH_MAPPING_TABLE.get(original_p, 'NOT_FOUND')
        data_tt = Util.get_timestamp_from_activelearning_time(date_str)

        
        video_path, csv_path = self.find_suitable_file_pair(data_tt, p, settings['remote_data_path'])

        print('video_path:', video_path)
        print('csv_path:', csv_path)

        self.current_video = video_path
        self.current_csv = csv_path

        self.settings['current_video'] = video_path
        self.settings['current_csv'] = csv_path
    

        print('self.settings:', self.settings)

        self.settings['current_video'] = self.current_video
        self.settings['current_csv'] = self.current_csv

        self.current_video_meta = VideoMetadata.get_meta(self.current_video)

        self.settings['fps'] = self.current_video_meta.fps

        print('New settings:')
        print(json.dumps(self.settings, sort_keys=True, indent=4))
        print()



        self.sensor_layout = ActiveLearningLayout(self.settings, self.time_asix_for_plot)
        # self.sensor_layout.addWidget(new QLabel())
        # self.timestamp

        self.max_frames = self.sensor_layout.get_len() - 1

        self.main_layout.addLayout(self.player_controls)
        self.main_layout.addLayout(self.labels_view)
        t_layout = QHBoxLayout()
        t_layout.addWidget(self.timestamp)
        self.main_layout.addLayout(t_layout)
        #self.main_layout.addChildWidget(self.timestamp)
        self.main_layout.addLayout(self.sensor_layout)

        self.slider.setRange(0, int(self.max_frames))

        self.time = self.sensor_layout.get_time()

        self.start_time = time.strptime(os.path.split(self.current_video)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI')
        

        # if args.i.endswith('.yaml'):
        #     ind_of_dvr=data_paths[0].index('DVR___')+6
        #     st_time=str(data_paths[0][ind_of_dvr:-4])


        self.timestamp.setText(time.strftime('    Timestamp:\n%Y-%m-%d %H:%M:%S', self.start_time))
        
        # 
        # if self.timeline.setup(self, self.sensor_layout.video, self.activelearning_output, self.current_video):
        #     self.timeline.show()
        # else:
        #     print('Module `Timeline` initialize fail.')
        # 
        # if self.timeline.csv_manager.get_data_count() > 0:
        #     self.timeline.on_cell_double_clicked(0, 0)
        # 

        #self.frame_buffer_widget.setup_data(self.activelearning_output.all_data, self.current_video, self.current_p)


        self.play_pause()

    active_learning_window = None

    def setup_al_window(self, al_window):
        self.active_learning_window = al_window
    
    def moveEvent(self, event):
        if self.active_learning_window != None:
            self.active_learning_window.attach_to(self)
        if self.timeline != None:
            self.timeline.attach_to(self)
        super().moveEvent(event)

    def import_from_config(self):
        self.setup_timeline(True)

    label_count = 0

    def setup_timeline(self, from_yaml=False, dataframe=pd.DataFrame([])):
        
        if from_yaml:
            self.activelearning_output = ActivelearningOutputManager(self.settings['activelearning_output'])
            if self.timeline.setup(self, self.sensor_layout.video, self.activelearning_output, self.current_video):
                self.timeline.show()
            else:
                print('Module `Timeline` initialize fail.')
        else:
            self.activelearning_output = ActivelearningOutputManager(None, dataframe)
            rc, cc = dataframe.shape
            self.label_count = rc
            if not dataframe.empty:
                if self.timeline.setup(
                    self, self.sensor_layout.video, 
                    self.activelearning_output,
                    self.current_video
                ):
                    self.timeline.show()
                else:
                    print('Module `Timeline` initialize fail.')
            else:
                print('dataframe is empty.')


        if self.timeline.csv_manager.get_data_count() > 0:
            self.timeline.on_cell_double_clicked(0, 0)


    def find_suitable_file_pair(self, time_t, p, base_dir):
        video_folder = base_dir + p + os.sep + 'Camera'
        csv_folder = base_dir + p + os.sep + 'Necklace'

        for video_file in os.listdir(video_folder):
            if not video_file.startswith('DVR___'): continue
            video_file_path = video_folder + os.sep + video_file
            video_file_time_t = Util.get_timestamp_from_video_file(video_file)
            video_file_end_time_t = video_file_time_t + Util.VideoMetadata.get_meta(video_file_path).get_video_length()

            if video_file_time_t <= time_t <= video_file_end_time_t:
                return video_file_path, csv_folder + os.sep + video_file.replace('DVR', FILE_FORMAT[:3].upper()).replace('AVI', FILE_FORMAT)
        
        return '', ''


    def get_frame_from_ts(self, start_ts, target_ts, fps):
        du = target_ts - start_ts
        return int(du.ms / 1000 * fps)


    def import_as_chewing(self):
        
        start_frame = self.get_frame_from_ts(
            Util.get_timestamp_from_video_file(self.current_video),
            self.segment_start_time,
            self.current_video_meta.fps
        )

        end_frame = self.get_frame_from_ts(
            Util.get_timestamp_from_video_file(self.current_video),
            self.segment_end_time,
            self.current_video_meta.fps
        )

        print('start_frame:', start_frame)
        print('end_frame:', end_frame)

        total_frame_count = end_frame - start_frame
        count = 0

        for f in range(start_frame, end_frame):
            if f in self.labels: count += 1
        
        if count / total_frame_count < 0.5:
            QMessageBox.question(self,'Alert','Import as chewing must tag half of frames.', QMessageBox.Yes, QMessageBox.Yes)
        else:
            self.label_count -= 1
            self.pending_task.append([self.activelearning_output.annotate, self.data_index, True])
            # self.activelearning_output.annotate(self.data_index, True)
            # self.timeline.reload(self.activelearning_output)
            
            self.timeline.u_timeline_table.setItem(self.data_index, 3, QTableWidgetItem('Chewing'))

            self.timeline.on_cell_double_clicked(self.data_index + 1, 0)

    pending_task = None

    def check_before_refine(self):
        for i in range(6):
            if self.timeline.u_timeline_table.item(i, 3) == None:
                return True
            if 'Not' in self.timeline.u_timeline_table.item(i, 3).text():
                return False
        
        return True

    def import_as_non_chewing(self):
        start_frame = self.get_frame_from_ts(
            Util.get_timestamp_from_video_file(self.current_video),
            self.segment_start_time,
            self.current_video_meta.fps
        )

        end_frame = self.get_frame_from_ts(
            Util.get_timestamp_from_video_file(self.current_video),
            self.segment_end_time,
            self.current_video_meta.fps
        )

        print('start_frame:', start_frame)
        print('end_frame:', end_frame)

        total_frame_count = end_frame - start_frame
        count = 0

        for f in range(start_frame, end_frame):
            if f in self.labels: count += 1
        
        if count != 0:
            QMessageBox.question(self,'Alert','Import as non-chewing must tag zero frame(s).', QMessageBox.Yes, QMessageBox.Yes)
        else:
            self.label_count -= 1
            self.pending_task.append([self.activelearning_output.annotate, self.data_index, False])
            # self.activelearning_output.annotate(self.data_index, False)
            # self.timeline.reload(self.activelearning_output)

            self.timeline.u_timeline_table.setItem(self.data_index, 3, QTableWidgetItem('Non-chewing'))

            self.timeline.on_cell_double_clicked(self.data_index, 0)

    def update_ui(self):

        self.w1.setXRange(self.frame - 20, self.frame + 20, padding=None)
        self.time_cursor.setValue(self.frame)


        self.current_index.setText('Frame:' + str(self.frame))

        new_time = datetime.fromtimestamp(time.mktime(self.start_time)) + timedelta(seconds=0)
        new_time += timedelta(seconds=self.frame / self.current_video_meta.fps)

        font = QFont()
        font.setBold(True)
        font.setPointSize(15)
        self.timestamp.setFont(font)
        self.timestamp.setText(new_time.strftime('    Timestamp: %Y-%m-%d %H:%M:%S'))

        self.update_slider(self.frame)

        self.sensor_layout.update_components(self.frame)

    def sliderPressed(self):
        self.flag_sliderPressed = True

    def slider_changed(self, frame):
        self.frame = frame
        self.current_index.setText('Frame:' + str(frame))
        
    def apply_slider_change(self):
        self.flag_sliderPressed = False
        self.update_ui()

    def _auto_load_next_video_handler(self):
        if self.frame >= self.max_frames:
            ts = Util.get_timestamp_from_video_file(self.current_video)
            video_folder = os.path.split(self.current_video)[0]

            map_table = []

            min_range = TimeT(99999999999999, 0)
            min_range_file = None

            for file_name in Util.listdir(video_folder):
                if file_name.startswith('DVR___'):
                    tt = Util.get_timestamp_from_video_file(video_folder + os.sep + file_name)
                    
                    if (tt - ts) > TimeT(0, 0) and (tt - ts) < min_range:
                        min_range = tt - ts
                        min_range_file = file_name
            
            if min_range_file == None:
                print('No more video file.')
                exit()

            self.current_video = video_folder + os.sep + min_range_file
            self.sensor_layout.video_reentrant(self.current_video)
            self.frame = 0
            self.max_frames = self.max_frames = self.sensor_layout.get_len() - 1
            self.slider.setRange(0, int(self.max_frames))

    video_stack = []

    def auto_load_next_video_handler(self):
        if self.frame > self.max_frames:
            video_folder = os.path.split(self.current_video)[0]
            video_file = os.path.split(self.current_video)[1]

            current_video_number = Util.get_video_number(video_file)

            found_videos = {}

            for file_name in Util.listdir(video_folder):
                video_number = Util.get_video_number(file_name)
                if video_number > current_video_number:
                    found_videos[video_number] = file_name
            
            keys = list(found_videos.keys())

            keys.sort()

            target_video = None

            if len(keys) > 0:
                target_video = found_videos[keys[0]]
            else:
                print('No more video file.')
                return 

            self.video_stack.append(video_folder + os.sep + video_file)

            print('Auto load next: from', video_file, 'to', target_video)

            self.start_time = time.strptime(os.path.split(target_video)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI')

            self.current_video = video_folder + os.sep + target_video
            self.sensor_layout.video_reentrant(self.current_video)
            self.frame = 0
            self.max_frames = self.max_frames = self.sensor_layout.get_len() - 1
            self.slider.setRange(0, int(self.max_frames))

            time_t = Util.get_timestamp_from_video_file(self.current_video)

            self.sensor_layout.reload_csv_by_video(self.current_video, self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)
            
            #self.sensor_layout.reload_csv(self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)

            #self.frame_buffer_widget.setup_data(self.activelearning_output.all_data, self.current_video, self.current_p)

    def auto_load_previous_video_handler(self):

        if self.frame < 0:
            print('Try to load last video.')
            print('self.current_video:', self.current_video)
            if False:
                print('Will load from stack.')
                previous_video = self.video_stack.pop(-1)
                self.current_video = previous_video
    
                time_t = Util.get_timestamp_from_video_file(self.current_video)

                meta = VideoMetadata.get_meta(self.current_video)

                video_length = meta.total_frame / meta.fps

                video_length -= 3

                video_length = video_length if video_length > 0 else 0

                t3 = TimeT(int(video_length), video_length - int(video_length))

                self.frame = meta.total_frame - 1

                self.sensor_layout.reload_csv(self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t - t3)

                #self.frame_buffer_widget.setup_data(self.activelearning_output.all_data, self.current_video, self.current_p)

                print('Auto load last: from', self.current_video, 'to', previous_video)

            else:
                print('Will load from folder.')
                video_folder = os.path.split(self.current_video)[0]
                video_file = os.path.split(self.current_video)[1]

                current_video_number = Util.get_video_number(video_file)

                found_videos = {}

                for file_name in Util.listdir(video_folder):
                    video_number = Util.get_video_number(file_name)
                    if video_number < current_video_number:
                        found_videos[video_number] = file_name
                
                keys = list(found_videos.keys())

                keys.sort()

                target_video = None

                if len(keys) > 0:
                    target_video = found_videos[keys[-1]]
                else:
                    print('No more video file.')
                    return 

                print('Auto load last: from', video_file, 'to', target_video)

                self.start_time = time.strptime(os.path.split(target_video)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI')

                self.current_video = video_folder + os.sep + target_video
                
                meta = VideoMetadata.get_meta(self.current_video)

                self.sensor_layout.video_reentrant(self.current_video)
                self.frame = meta.total_frame - 1
                self.max_frames = self.max_frames = self.sensor_layout.get_len() - 1
                self.slider.setRange(0, int(self.max_frames))

                time_t = Util.get_timestamp_from_video_file(self.current_video)

                self.sensor_layout.reload_csv_by_video(self.current_video, self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)
                # self.sensor_layout.reload_csv(self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)

                #self.frame_buffer_widget.setup_data(self.activelearning_output.all_data, self.current_video, self.current_p)

    def on_frame_changed(self):
        self.auto_load_previous_video_handler()
        self.auto_load_next_video_handler()
        #self.frame_buffer_widget.update(self.frame)
        self.update_ui()

    def timer_update(self):
        self.frame = self.frame + 1
        self.on_frame_changed()

    def update_slider(self, frame):
        if not self.flag_sliderPressed: self.slider.setValue(frame)

    def do_play_pause(self):
        if self.playing:
            self.update_timer.stop()
        else:
            self.update_timer.start(int(1000 / self.current_video_meta.fps))
        self.playing = ~self.playing

    def _pause(self):
        self.playing = False
        self.update_timer.stop()

    def play_pause(self):
        self.do_play_pause()
        self.update_play_status_ui()
        #self.grab().save('exported_images' + os.sep + str(int(time.time())) + '.png')

    def update_play_status_ui(self):
        if self.playing: self.play_pause_btn.setText('Pause')
        else: self.play_pause_btn.setText('Play')

    def keyPressEvent(self, e):
        if e.key() == 65:
            # Moving left
            # self.update_slider(self.frame - 1)
            self._pause()
            self.play_pause_btn.setText('Play')
            self.frame -= 1
            self.on_frame_changed()
            
        elif e.key() == 68:
            # Moving right
            #self.update_slider(self.frame + 1)
            self._pause()
            self.play_pause_btn.setText('Play')
            self.frame += 1
            self.on_frame_changed()

        if e.key() == 83:
            self.labeling = ~self.labeling
            if self.labeling:
                self.labeling_l.setText('Labeling')
            else:
                self.labeling_l.setText('')
        if self.labeling or e.key() == 87:
            # Labeling butto n pressed
            if self.frame not in self.labels:
                self.labels += [self.frame]
                spots = [{'pos': [self.frame, 1], 'data': 1}]
                self.s1.addPoints(spots)
            else:
                self.labels.remove(self.frame)
                self.repaint_labels()

    def clicked(self, plot, points):
        self.lastClicked
        for p in self.lastClicked:
            p.resetPen()
        self.update_slider(points[-1].pos()[0])
        for p in points:
            p.setPen('b', width=2)
        self.lastClicked = points

    def repaint_labels(self):
        spots = [{'pos': [l, 1], 'data': 1} for l in self.labels]
        self.s1.clear()
        self.s1.addPoints(spots)

    def dragged(self, e):
        e.setValue(np.ceil(e.value()))
        self.update_slider(e.value())

    def import_labels(self):
        if self.label_path != None:
            print('self.label_path:', self.label_path)
            
            read_fn = None
            if FILE_FORMAT == 'xlsx': read_fn = pd.read_excel
            else: read_fn = pd.read_csv
            self.loaded_labels = read_fn(self.label_path, index_col=0)
            
            self.labels = self.loaded_labels.copy()
            self.labels = self.labels.dropna()
            self.labels = self.labels.index.values.tolist()
            self.repaint_labels()
        else:
            print('still working on this feature. In the meanwhile you can specify it using the yaml settings file')

    def export_labels(self):

        file_name = time.strftime("export_%Y-%m-%d_%H_%M_%S.csv", time.localtime()) 
        file_path = 'export' + os.sep + file_name

        self.activelearning_output.all_data.to_csv(file_path, header=False)

        print('File exported to', file_path)

        #output_file = self.label_path
        #
        #if os.path.exists(output_file):
        #    save_timestamp = str(int(datetime.now().timestamp()))
        #    print('Label file exist. Creating another one', save_timestamp)
        #    output_file = output_file[:-4] + save_timestamp + '.csv'
        #pd.DataFrame(self.time).iloc[self.labels].to_csv(output_file, header=True)
        #self.time.iloc[self.labels].to_csv(output_file, header=True)

        # if self.loaded_labels is None:
        #     self.time.iloc[self.labels].to_csv(output_file, header=True)
        # else:
        #     self.loaded_labels['updated'] = None
        #     self.loaded_labels.loc[self.labels, 'updated'] = 1
        #     self.loaded_labels.to_csv(output_file, header=True)

    def find_video_by_ts(self, folder, time_t):
        folder += (os.sep + 'Camera')
        files = Util.listdir(folder)
        
        print('Try to find video with target time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_t.s)))

        for file_path in files:

            if not file_path.startswith('DVR___'): continue
            
            ts = Util.get_timestamp_from_video_file(file_path)

            meta = Util.VideoMetadata(folder + os.sep + file_path)

            ts_end = ts + TimeT(
                meta.total_frame // meta.fps,
                meta.total_frame / meta.fps - int(meta.total_frame / meta.fps)
            )

            print('    File:', file_path)
            print('    start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts.s)))
            print('      end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_end.s)))
            print()

            if ts <= time_t <= ts_end:
                    return folder + os.sep + file_path

        
        return None

    def navigate_to(self, p, time_t, time_t_end, data_index):

        print('::navigate_to:', p, time_t, time_t_end)

        self.segment_start_time = time_t
        self.segment_end_time = time_t_end
        self.data_index = data_index

        print('self.segment_start_time:', self.segment_start_time)

        if self.mode == 'remote_data_path':

            self.current_p = p
            # print(p, time_t)

            folder = self.settings['remote_data_path'] + PATH_MAPPING_TABLE[self.current_p]

            print('target folder:', folder)

            target_video = self.find_video_by_ts(folder, time_t)
            print('target_video:', target_video)
            if target_video == None:
                print('Can\'t find suitable video file.')
                QMessageBox.question(self,'Alert','Can\'t find suitable video file. Program will not work.', QMessageBox.Yes, QMessageBox.Yes)
                return
            
            
            self.start_time = time.strptime(os.path.split(target_video)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI')

            target_video_meta = Util.VideoMetadata(target_video)

            # - video_start + time_t
            target_video_frame =                                                  \
                (time_t.ms - Util.get_timestamp_from_video_file(target_video).ms) \
                    /                                                             \
                (target_video_meta.total_frame / target_video_meta.fps * 1000)    \
                    *                                                             \
                target_video_meta.total_frame
            
            target_video_frame = int(target_video_frame)

            self.frame = target_video_frame

            self.current_video = target_video

            
            self.sensor_layout.video_reentrant(target_video)
            # self.sensor_layout.reload_csv(self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)
            self.sensor_layout.reload_csv_by_video(self.current_video, self.settings['remote_data_path'], PATH_MAPPING_TABLE[self.current_p], time_t)

            self.update_ui()

        #self.frame_buffer_widget.setup_data(self.activelearning_output.all_data, self.current_video, self.current_p)

    def reject(self):
        exit(0)




if __name__ == '__main__':
    app = QApplication([])

    args = parser.parse_args()

    active_learning = ActiveLearning()
    
    timeline = Timeline()
    window = SenseView(args, timeline)
    window.show()
    window.timeline.attach_to(window)
    active_learning.attach_to(window)
    window.setup_al_window(active_learning)
    active_learning.show()
    window.activateWindow()
    app.exit(app.exec_())

