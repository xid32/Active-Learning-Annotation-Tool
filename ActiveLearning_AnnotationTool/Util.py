

import os, string, time
import cv2

class TimeT:

    _s = 0
    _ms = 0.0

    def __init__(self, s, ms):
        self._s = int(s)
        self._ms = float(ms)
    
    @property
    def s(self):
        return self._s

    @property
    def ms(self):
        return int((self._s + self._ms) * 1000)
    
    @property
    def pos(self):
        return self._ms
    
    def __str__(self):
        return f'<class {self.__class__.__name__} {self.ms}>'

    def __eq__(self, obj):
        return self.s == obj.s

    def __lt__(self, obj):
        return self.ms < obj.ms

    def __le__(self, obj):
        return self.ms <= obj.ms

    def __sub__(self, obj):
        return TimeT(
            self._s - obj._s, 
            self._ms - obj._ms
        )

    def __add__(self, obj):
        return TimeT(
            self._s + obj._s, 
            self._ms + obj._ms
        )


video_meta_cache = {}

class VideoMetadata:

    fps = 0
    total_frame = 0

    @classmethod
    def get_meta(clazz, file_path):
        global video_meta_cache
        if file_path not in video_meta_cache:
            video_meta_cache[file_path] = clazz(file_path)
        
        return video_meta_cache[file_path]

    def get_video_length(self):
        tinsd = self.total_frame / self.fps
        tins = int(tinsd)
        tinms = tinsd - tins
        return TimeT(tins, tinms)

    def __init__(self, file_path):
        capture = cv2.VideoCapture(file_path)
        self.total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()



def get_timestamp_from_video_file(file_path):
    # DVR___2019-01-28_12.14.36.AVI
    file_name = os.path.split(file_path)[1].upper()
    return TimeT(
        time.mktime(time.strptime(os.path.split(file_name)[1], 'DVR___%Y-%m-%d_%H.%M.%S.AVI')),
        0
    )

# 2019-01-12 13:07:04-06:00
# 2019-01-28 22:19:06.600000-06:00
def get_timestamp_from_activelearning_time(time_string):
    if '.' in time_string:
        parts = time_string.split('.')
        dt_str = parts[0]
        ms_str = '0.' + parts[1].split('-')[0]
        return TimeT(
            time.mktime(time.strptime(dt_str, '%Y-%m-%d %H:%M:%S')),
            ms_str
        )
    else:
        return get_timestamp_from_csv_data(time_string)


# 2019-01-24 12:34:55-06:00
def get_timestamp_from_csv_data(time_string):
    dt_str = time_string[:-len('-06:00')]
    return TimeT(
        time.mktime(time.strptime(dt_str, '%Y-%m-%d %H:%M:%S')),
        0
    )

#1548338303420
def get_timestamp_from_ms(ms):
    ms_str = str(ms)
    return TimeT(
        ms_str[:-3],
        '0.' + ms_str[-3:]
    )

def get_timestamp_from_s(s):
    return TimeT(str(s), 0)

def listdir(path):
    pre_result = os.listdir(path)
    pre_result.sort()
    return pre_result

def get_video_number(file_name):
    number_string = ''

    for ch in file_name:
        if ch in string.digits:
            number_string += ch
    if len(number_string) == 0:
        return 4294967295
    return int(number_string)

