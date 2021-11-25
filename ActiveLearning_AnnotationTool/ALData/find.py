

import pandas as pd

import os


for file_name in os.listdir('.'):
    print(file_name)
    if file_name.endswith('.xlsx'):
        df = pd.read_excel(file_name)
        print('    ', df[df.start == '2019-01-24 11:54:52.750000-06:00'])






