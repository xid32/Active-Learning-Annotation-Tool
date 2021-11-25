import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yaml
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QMainWindow, QApplication, QSlider, QLabel

import pyqtgraph as pg

import argparse

from Layouts import *

from Timeline import Timeline

import csv

import traceback

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True, help='path to input (image folder or CSV)')
parser.add_argument('-l', type=str, required=True, help='layout name')


PATH_MAPPING_TABLE = {
    "P101":"P203-2",
    "P102":"P205",
    "P103":"P206",
    "P104":"P207",
    "P105":"P209",
    "P106":"P210",
    "P107":"P211",
    "P108":"P212",
    "P109":"P217",
    "P110":"P218"
}


class Color(QWidget):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class SenseView(QMainWindow):
    frame = 0
    max_frames = 20000
    current_video_frame_number = 0
    timeline = None

    def reentrant(self):
        pass

    def findPath(self,folder,timeStr,user):

        print(self,folder,timeStr,user)

        s1=timeStr[5:10]
        s2=timeStr[2:4]
        s3=timeStr[11:13]

        #get the rawdata file path
        filePath=os.path.join(folder,"Necklace")
        filePath=os.path.join(filePath,s1+"-"+s2+"_"+s3+".csv")

        #check if the file exists and output file path
        if os.path.exists(filePath):
            print("...find rawdata file path: "+filePath)
        else:
            print("not find rawdata file")
        
        #get the vedio file path
        path=os.path.join(folder,"Camera")
        #get the start time string
        timeStr=timeStr[0:len(timeStr)-13]

        #get the start time
        startTime=datetime.strptime(timeStr,"%Y-%m-%d %H:%M:%S")
        minDurn=-1
        fileName=None
        
        print(folder)

        #get the file list in the path
        fileList=os.listdir(path)    
        
        #loop to find the closest file name
        for name in fileList:                
            tempPath=os.path.join(path,name)
            #if the file is a directory
            if os.path.isdir(tempPath):
                continue

            s=name[6:]
            s=s[0:len(s)-4]
            #get file time from file name
            try:
                fileTime = datetime.strptime(s,"%Y-%m-%d_%H.%M.%S")
            except:
                continue
            durn=(fileTime-startTime).total_seconds()
            
            #get the difference of file time and start time
            if durn>=0:
                continue
            
            durn=abs(durn)
            
            #compare the difference of minDurn and durn
            if minDurn==-1:
                minDurn=durn
                fileName=name
            elif minDurn>durn:
                minDurn=durn
                fileName=name
        
        #check the fileName and output
        fileName=os.path.join(path,fileName)
        if fileName:
            print("find correct video file path: "+fileName)
        else:
            print("not find correct video file")
        print()

        return filePath,fileName



    def __init__(self, args, timeline):
        super().__init__()

        self.timeline = timeline

        self.window = QWidget()
        self.setCentralWidget(self.window)
        self.main_layout = QVBoxLayout()
        self.window.setLayout(self.main_layout)

        ########  adding the player controls, slider and timer ########
        # Play and pause button
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setText("P")
        self.playing = False
        self.play_pause_btn.clicked.connect(self.play_pause)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)

        # adding slider and player to a layout
        self.player_controls = QHBoxLayout()
        self.player_controls.addWidget(self.play_pause_btn)
        self.player_controls.addWidget(self.slider)

        self.main_layout.addLayout(self.player_controls)

        # setting up a timer to play the video automatically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.timer_update)

        ########   adding annotation related code ########

        self.labels_view = QHBoxLayout()
        self.labeling = False

        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        view.setMaximumHeight(100)
        self.labels_view.addWidget(view)

        # max_x = self.max_frames
        self.w1 = view.addPlot()
        # w1.setLimits(yMin=0,yMax=3,minYRange=0,maxYRange=3)
        self.w1.setMouseEnabled(y=False)
        self.w1.setXRange(0, 50, padding=None)
        self.w1.setYRange(0, 3, padding=None)
        ax_b = self.w1.getAxis('bottom')  # This is the trick
        # ax_b.setTicks([])
        # ax_l = w1.getAxis('left')  # This is the trick
        # ax_l.setHeight(100)
        self.main_layout.addLayout(self.labels_view)

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
        self.labeling_l.setText("")
        import_export_view.addWidget(self.labeling_l)


        self.timestamp = QLabel()
        self.timestamp.setText("Timestamp:")
        import_export_view.addWidget(self.timestamp)

        self.current_index = QLabel()
        self.current_index.setText("Frame:0")
        import_export_view.addWidget(self.current_index)

        import_btn = QPushButton()
        import_btn.setText("import")
        import_btn.clicked.connect(self.import_labels)
        import_export_view.addWidget(import_btn)

        export = QPushButton()
        export.setText("export")
        export.clicked.connect(self.export_labels)
        import_export_view.addWidget(export)

        self.labels_view.addLayout(import_export_view)

        data_paths = None
        self.label_path = None

        activelearning_output_file = None

        ######## User defined sensor layout #######
        if args.i.endswith(".yaml"):
            with open(args.i) as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)

                if settings["activelearning_output"]==None or settings["remote_data_path"]==None:

                    data_paths = settings["data_paths"]

                    if settings["activelearning_output"] != None:
                        activelearning_output_file = settings["activelearning_output"][0]
                    

                else:
                     
                    dict={"P101":"P203-2","P102":"P205","P103":"P206","P104":"P207",
                          "P105":"P209","P106":"P210",
                          "P107":"P211","P108":"P212","P109":"P217","P110":"P218"}
                    
    
                    file = open(settings["activelearning_output"][0], "r")


                    activelearning_output_file = settings["activelearning_output"][0]
                     
                    #read header line
                    file.readline()
                    cwd=os.getcwd()

                
                    #loop to read lines and find path
                    while True:
                        line = file.readline()
                
                        if line:
                            line=line.strip()
                            arr=line.split(",")
                            #get the corresponding folder
                            #folder=os.path.join(cwd,dict[arr[0]])
                            folder=dict[arr[0]]

                
                
                            #find path
                            filePath,fileName=self.findPath(settings["remote_data_path"]+folder,arr[1],dict[arr[0]])
                             
                            data_paths=[fileName,filePath]
                            #data_paths = settings["remote_data_path"]
                            #settings = {"data_paths": [fileName,filePath]}
                            settings["data_paths"] = data_paths
                            print(data_paths)
                            print("\n")
                            #print(settings)
                
                        else:  
                            break
                
                    file.close()

                self.label_path = settings["label_path"]
        else:
            settings = {"data_paths": [args.i]}

        print(settings)
        print(args.l + "." + args.l + "(settings)")
        cmd_str = args.l + "." + args.l + "(settings)"
        self.sensor_layout = eval(cmd_str)
        self.max_frames = self.sensor_layout.get_len() - 1
        self.main_layout.addLayout(self.sensor_layout)

        print(self.max_frames)
        self.slider.setRange(0, self.max_frames)

        self.time = self.sensor_layout.get_time()
        

        if args.i.endswith(".yaml"):
            ind_of_dvr=data_paths[0].index('DVR___')+6

            
            st_time=str(data_paths[0][ind_of_dvr:-4])
            #2019-01-28_12.14.36

            self.start_time= datetime.strptime(st_time, '%Y-%m-%d_%H.%M.%S')


            
        else:
            self.start_time= datetime.strptime(str(self.time[0]), '%Y-%m-%dT%H:%M:%S:%f')

         
        self.timestamp.setText("" + str(self.start_time))

        if self.timeline.setup(self, self.sensor_layout.video, activelearning_output_file, data_paths[0]):
            self.timeline.show()


        self.settings = settings

        if os.path.isdir(self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE')):
            print('`remote_data_path` found, switch to remote data path mode.')
            csv_file = open(self.settings['activelearning_output'][0], 'r')
            reader = csv.reader(csv_file)

            target_data = None
            folder = None
            for line in reader:
                folder = PATH_MAPPING_TABLE.get(line[0], 'NOT_FOUND')
                if folder == 'NOT_FOUND': continue
                target_data = line
                break

            csv_file.close()
            
            if target_data != None:
                relocated_time = target_data[1]
                relocated_datetime = relocated_time.split('.')[0]
                relocated_timestamp = int(time.mktime(time.strptime(relocated_datetime, '%Y-%m-%d %H:%M:%S')))
                thousandths = float('0.' + relocated_time.split('.')[1].split('-')[0])
                selected_video_file = None

                folder_base = self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE') + folder + os.sep + 'Camera'

                for file_index in range(len(os.listdir(folder_base)) - 1):

                    file_start = os.listdir(folder_base)[file_index]
                    file_end = os.listdir(folder_base)[file_index + 1]

                    datetime_part_with_ext_name_start = file_start.split('___')[-1]
                    datetime_part_start = '.'.join(datetime_part_with_ext_name_start.split('.')[:-1])
                    start_time = int(time.mktime(time.strptime(datetime_part_start, '%Y-%m-%d_%H.%M.%S')))

                    datetime_part_with_ext_name_end = file_end.split('___')[-1]
                    datetime_part_end = '.'.join(datetime_part_with_ext_name_end.split('.')[:-1])
                    end_time = int(time.mktime(time.strptime(datetime_part_end, '%Y-%m-%d_%H.%M.%S')))

                    if start_time <= relocated_timestamp < end_time:
                        selected_video_file = os.listdir(folder_base)[file_index]
                        break
            
                ymd = selected_video_file.split('_')[3]

                y = int(ymd.split('-')[0])
                m = int(ymd.split('-')[1])
                d = int(ymd.split('-')[2])

                csv_prefix = str(m).zfill(2) + '-' + str(d).zfill(2) + '-' + str(y)[-2:]

                print(csv_prefix)
                
                csv_folder_base = self.settings.get('remote_data_path', 'FIELD_NOT_AVAILABLE') + folder + os.sep + 'Necklace'

                relocated_timestamp_thousandths = (relocated_timestamp + thousandths) * 1000

                target_csv_file_name = None

                for file_name in os.listdir(csv_folder_base):

                    if file_name.startswith(csv_prefix):
                        
                        file_path = csv_folder_base + os.sep + file_name
                        csv_file = open(file_path, 'r')
                        reader = csv.reader(csv_file)
                        line_list = list(reader)

                        for line_index in range(len(line_list) - 1):
                            try:
                                line_h = line_list[line_index][0]
                                line_t = line_list[line_index + 1][0]

                                if float(line_h) <= relocated_timestamp_thousandths <= float(line_t):
                                    target_csv_file_name = file_name
                                    break
                            except:
                                traceback.print_exc()
                
                print(target_csv_file_name)




    def play_record():
        pass




    def slider_changed(self, frame):
        self.frame = frame
        self.w1.setXRange(self.frame - 20, self.frame + 20, padding=None)
        self.time_cursor.setValue(self.frame)
        self.current_index.setText("Frame:" + str(frame))
        new_time= self.start_time + timedelta(seconds=0)
        new_time+= timedelta(seconds=frame/5)
        new_time=str(new_time)
        if len(new_time)<22:
            new_time+=":00"
        elif len(new_time)>22:
            new_time=new_time[0:22]
            new_time=new_time.replace('.',':')

        self.timestamp.setText("" + new_time)


        self.update_components()

    def timer_update(self):
        self.frame = self.frame + 1
        self.update_slider(self.frame)

    def update_slider(self, frame):
        self.slider.setValue(frame)

    def play_pause(self):
        if self.playing:
            self.update_timer.stop()
        else:
            self.update_timer.start(100)
        self.playing = ~self.playing

    def keyPressEvent(self, e):
        if e.key() == 65:
            # Moving left
            self.update_slider(self.frame - 1)
        elif e.key() == 68:
            # Moving right
            self.update_slider(self.frame + 1)

        if e.key() == 83:
            self.labeling = ~self.labeling
            if self.labeling:
                self.labeling_l.setText("Labeling")
            else:
                self.labeling_l.setText("")

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
            print(self.label_path)
            self.loaded_labels = pd.read_csv(self.label_path, index_col=0)
            self.labels = self.loaded_labels.copy()
            self.labels = self.labels.dropna()
            self.labels = self.labels.index.values.tolist()
            self.repaint_labels()
        else:
            print("still working on this feature. In the meanwhile you can specify it using the yaml settings file")

    def export_labels(self):

        output_file = self.label_path

        if os.path.exists(output_file):
            save_timestamp = str(int(datetime.now().timestamp()))
            print("Label file exist. Creating another one", save_timestamp)
            output_file = output_file[:-4] + save_timestamp + ".csv"
        pd.DataFrame(self.time).iloc[self.labels].to_csv(output_file, header=True)
        #self.time.iloc[self.labels].to_csv(output_file, header=True)

        # if self.loaded_labels is None:
        #     self.time.iloc[self.labels].to_csv(output_file, header=True)
        # else:
        #     self.loaded_labels["updated"] = None
        #     self.loaded_labels.loc[self.labels, "updated"] = 1
        #     self.loaded_labels.to_csv(output_file, header=True)

    def update_components(self):
        self.sensor_layout.update_components(self.frame)


if __name__ == '__main__':
    app = QApplication([])

    args = parser.parse_args()

    print(args)
    timeline = Timeline()
    window = SenseView(args, timeline)
    window.show()
    app.exit(app.exec_())