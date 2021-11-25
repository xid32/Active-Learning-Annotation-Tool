



import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yaml
from PyQt5.QtCore import QPoint, QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QMainWindow, QApplication, QSlider, QLabel

import pyqtgraph as pg

import argparse

from Layouts import *

from Timeline import Timeline

import csv
import signal
import sys
import traceback

import json

import Util
from Util import VideoMetadata

from ActivelearningOutputManager import ActivelearningOutputManager

from Config import FILE_FORMAT


def findPath(self,folder,timeStr,user):

        print(self,folder,timeStr,user)

        s1=timeStr[5:10]
        s2=timeStr[2:4]
        s3=timeStr[11:13]

        #get the rawdata file path
        filePath=os.path.join(folder,"Necklace")
        filePath=os.path.join(filePath,s1+"-"+s2+"_"+s3+"." + FILE_FORMAT)

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
        fileList=Util.listdir(path)    
        
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
