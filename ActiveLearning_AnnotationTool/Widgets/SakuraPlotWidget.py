
from pyqtgraph import PlotWidget, InfiniteLine, mkPen, mkColor
import numpy as np
import os
import pandas as pd
import time

import traceback

from Util import *

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient, QPen)
from PyQt5.QtWidgets import *

import os

from Config import FILE_FORMAT

try:
    os.mkdir('exported_images')
except:
    pass

class SakuraPlotWidget(PlotWidget):

    last_index = 0

    Y_RANGE_MIN = 100
    Y_RANGE_MAX = 5000

    current_y_range_min = 2600
    current_y_range_max = 4600

    flag_csv_reloaded = True

    is_lazy_init_called = False

    fps = 0

    tick_marks = None

    def __init__(self, data_path, time_axis):
        super().__init__(axisItems={'bottom': time_axis})

        print('data_path:', data_path)

        # if data_path == None: return

        self.tick_marks = []

        self.lazy_init(data_path)

        

    mi = 0
    ma = 0

    def draw_tick_mark_tears_down(self):
        return 
        for item in self.tick_marks:
            try:
                self.removeItem(item)
            except:
                pass
        self.tick_marks.clear()

    def draw_tick_mark(self):
        return
        self.draw_tick_mark_tears_down()
        fps = -1
        if self.fps != 0: fps = self.fps
        else: fps = 20

        draw_every_n_sec = 5

        draw_every_n_frame = draw_every_n_sec * fps

        current = 0

        while current < len(self.sensor_data.index):
            self.addItem(InfiniteLine(pos=current, angle=90, pen=mkPen(color=mkColor('#00FF00'))))
            current += draw_every_n_frame


    def setYRange_wrapper(self, mi, ma):
        self.mi = mi
        self.ma = ma
        #print('Y Range will be setted to', mi, ma)
        self.setYRange(mi, ma)

    def lazy_init(self, data_path):

        # if self.is_lazy_init_called: return 
        # self.is_lazy_init_called = True
        
        target_csv_files_folder = data_path

        print('SakuraPlotWidget::target_csv_files_folder:', target_csv_files_folder)

        li = []

        read_fn = None
        if FILE_FORMAT == 'xlsx': read_fn = pd.read_excel
        else: read_fn = pd.read_csv

        if os.path.isdir(data_path):
            for file_name in os.listdir(target_csv_files_folder):

                if file_name.endswith('.' + FILE_FORMAT):

                    df = read_fn(target_csv_files_folder + os.sep + file_name)
                    li.append(df)
        else:
            df = read_fn(data_path)
            li.append(df)

        self.sensor_data = pd.concat(li, axis=0, ignore_index=True)

        self.data_str_convert()


        self.sensor_data = self.sensor_data.sort_values('date')

        print('self.sensor_data.values[:, 2]:', self.sensor_data.values[:, 2])

        self.p1 = self.plot(self.sensor_data.values[:, 2])
        self.p1.setPen((255, 255, 0))


        print('self.p1:', self.p1)

        self.setYRange_wrapper(-1, 1)
        self.setMaximumHeight(400)
        self.setMouseEnabled(y=False)
        #self.ax_b = self.getAxis('bottom')
        #self.ax_b .setTicks([])

        self.time_cursor = InfiniteLine(angle=90)
        self.addItem(self.time_cursor)

        self.viewbox_size = 100

        self.setup_range()

        self.compile_ranger()

        self.draw_tick_mark()

    def compile_ranger(self):
        proximities = self.sensor_data['proximity'].tolist()
        STEP = 300
        ranges = []
        for i, v in enumerate(proximities):
            start = 0 if i - STEP < 0 else i - STEP
            end = len(proximities) - 1 if i + STEP >= len(proximities) else i + STEP
            segment = proximities[start:end]
            ranges.append((min(segment) - 150, max(segment) + 150))
        self.ranges = ranges

    def data_str_convert(self):
        def convert(x):
            if type(x) == int: return x
            if '.' in x: return get_timestamp_from_activelearning_time(x).ms
            return get_timestamp_from_csv_data(x).ms
        self.sensor_data['date'] = self.sensor_data['date'].apply(convert)

    def setup_range(self, base_index=0):
        return 
        value_min = self.sensor_data.min()['proximity']
        value_max = -1

        for i in range(-100, 100):
            try:
                proximity = self.sensor_data.loc[base_index + i]['proximity']
                if proximity > value_max: value_max = proximity
            except:
                pass

        self.value_min = value_min - 500
        self.value_max = value_max + 500

        self.setYRange_wrapper(self.value_min, self.value_max)

        # print('self.value_min:', self.value_min)
        # print('self.value_max:', self.value_max)

    def setup(self, start_ts, fps, total_frame):
        self.start_ts = start_ts
        self.fps = fps
        self.total_frame = total_frame

    def get_len(self):
        return self.sensor_data.shape[0]

    last_frame = 0

    def update_by_frame(self, frame_index):
        self.last_frame = frame_index
        target_ts = self.start_ts + frame_index / self.fps
        target_ts *= 1000

        if False:
            print('frame_index:', frame_index)
            print('total_frame:', self.total_frame)
            print('fps:', self.fps)
            print('target_ts:', target_ts)
            print()
            print()

        # print(self.sensor_data['date'], target_ts)
        query = self.sensor_data[self.sensor_data['date'] > target_ts]

        target_index = query.index.tolist()[0]

        
        draw_info = time.strftime('Draw: %Y-%m-%d %H:%M:%S', time.localtime(self.sensor_data.loc[target_index]['date'] / 1000))
        draw_info += '.' + str(int(self.sensor_data.loc[target_index]['date']) % 1000)
        draw_info += ' '
        draw_info += str(self.sensor_data.loc[target_index]['proximity'])
        # print(draw_info)

        self.update(target_index)

    def update(self, index):
        if False:
            print('SakuraPlotWidget::update ->', index)
        self.disableAutoRange()
        self.time_cursor.setValue(index)
        self.setXRange(index-self.viewbox_size, index+self.viewbox_size)


        if self.flag_csv_reloaded:
            self.setup_range(index)

        #print('Current proximity:', self.sensor_data.loc[index]['proximity'])
        
        if False:
            for r in [100, 10, 5, 1, 0]:
                try:
                    proximity = self.sensor_data.loc[index + r]['proximity']
                    
                    if proximity > self.value_max:
                        self.value_max = proximity + 1500
                        print('Range changed.')
                        print('self.value_min:', self.value_min)
                        print('self.value_max:', self.value_max)
                        print()

                    
                    if proximity < self.current_y_range_min:
                        self.current_y_range_min = (proximity - 100) if (proximity - 100) > 0 else 0

                    self.setYRange_wrapper(self.value_min, self.value_max + 2500)
                    break
                except:
                    traceback.print_exc()
        
        self.setYRange_wrapper(*self.ranges[index])


    def on_frame_change(self, frame):
        self.update_by_frame(frame + 2)


    def get_time(self):
        return self.sensor.get_time()

    def reload_csv_by_video(self, video_file, base_dir, p, time_t):
        full_filename = os.path.split(video_file)[1]
        filename = os.path.splitext(full_filename)[0]

        self.flag_csv_reloaded = True
        print('reload_csv_by_video:', video_file, base_dir, p, time_t)

        file_prefix = filename

        li = []

        csv_dir_path = base_dir + p + os.sep + 'Necklace'

        print('csv_dir_path parts:', base_dir, p, os.sep, 'Necklace')

        print('file_prefix:', file_prefix)

        file_prefix = file_prefix.replace('DVR', FILE_FORMAT[:3].upper())

        # P206 Necksense_Freeliving_Subset/ / Necklace

        for file_name in os.listdir(csv_dir_path):
            if file_name.startswith(file_prefix):
                print('reload_csv_by_video: file', file_name, 'loaded.')

                read_fn = None
                if FILE_FORMAT == 'xlsx': read_fn = pd.read_excel
                else: read_fn = pd.read_csv

                df = read_fn(csv_dir_path + os.sep + file_name)
                li.append(df)
        print('li:', li)
        if len(li) > 0:
            self.sensor_data = pd.concat(li, axis=0, ignore_index=True)
            self.data_str_convert()
            self.sensor_data = self.sensor_data.sort_values('date')
        else:
            print('Can\'t find csv for', file_prefix, 'in', csv_dir_path, '*')

        self.setup_range()

        self.setYRange_wrapper(0, 100)

        self.removeItem(self.p1)
        self.p1 = self.plot(self.sensor_data.values[:, 2])
        self.p1.setPen((255, 255, 0))

        self.compile_ranger()

        self.draw_tick_mark()

        print('::reload_csv_by_video() END')

    def reload_csv(self, p, base_dir, time_t):
        self.flag_csv_reloaded = True
        print('reload_csv:', base_dir, p, time_t)
        file_prefix = time.strftime('%m-%d', time.localtime(time_t.s))

        li = []

        csv_dir_path = base_dir + p + os.sep + 'Necklace'

        # if not self.flag_csv_reloaded: self.lazy_init(csv_dir_path)

        print('file_prefix:', file_prefix)

        for file_name in os.listdir(csv_dir_path):
            if file_name.startswith(file_prefix):
                print('reload_csv: file', file_name, 'loaded.')

                read_fn = None
                if FILE_FORMAT == 'xlsx': read_fn = pd.read_excel
                else: read_fn = pd.read_csv


                df = read_fn(csv_dir_path + os.sep + file_name)
                li.append(df)

        if len(li) > 0:
            self.sensor_data = pd.concat(li, axis=0, ignore_index=True)
            self.data_str_convert()
            self.sensor_data = self.sensor_data.sort_values('date')
        else:
            print('Can\'t find csv for', file_prefix, 'in', csv_dir_path, '#')

        self.setup_range()

        self.grab().save('exported_images' + os.sep + str(int(time.time())) + '.png')

        self.setYRange_wrapper(0, 100)

        self.compile_ranger()

        self.removeItem(self.p1)
        self.p1 = self.plot(self.sensor_data.values[:, 2])
        self.p1.setPen((255, 255, 0))

        self.draw_tick_mark()


        print('::reload_csv() END')


    def wheelEvent(self, event):
        super(SakuraPlotWidget, self).wheelEvent(event)
        self.viewbox_size = np.ceil((self.ax_b.range[1]-self.ax_b.range[0])/2)
