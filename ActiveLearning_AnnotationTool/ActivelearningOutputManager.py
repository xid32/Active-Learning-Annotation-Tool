

import os
import traceback

import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem

from Util import get_timestamp_from_activelearning_time


from ActiveLearning import dataset_folder

class ActivelearningOutputManager:

    all_data = None

    def __init__(self, csv_files, data_frame=pd.DataFrame([])):
        if data_frame.empty:
            dfs = []
            for file_path in csv_files:
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path, header=None)
                    dfs.append(df)

            self.all_data = pd.concat(dfs)
        else:
            self.all_data = data_frame

        # self.all_data.sort_values([0, 1])

        if self.all_data.shape[1] < 4:
            self.all_data[len(self.all_data.columns)] = -1

    def annotate(self, index, is_chewing):
        global dataset_folder

        value = 1 if is_chewing else 0
        self.all_data.iat[index, 3] = value

        p = self.all_data.iat[index, 0].lower().strip()
        ts = self.all_data.iat[index, 1].strip()
        te = self.all_data.iat[index, 2].strip()

        print('AOM::annotate:', p, ts, te)

        
        p_filename = f'feature-{p}.xlsx'

        if dataset_folder == None:
            dataset_folder = 'ALData'

        p_filepath = dataset_folder + os.sep + p_filename

        print(p_filepath, 'exist:', os.path.isfile(p_filepath))

        
        df = pd.read_excel(p_filepath)
        selected_index = df[(df.start == ts) & (df.end == te)].index
        
        print('selected_index:', selected_index)

        final_index = selected_index[0]

        print('final_index:', final_index)

        value_to_set = 0
        if is_chewing: value_to_set = 1

        df.iat[int(final_index), -4] = value_to_set

        df.to_excel(p_filepath)

    def get_data_count(self):
        return self.all_data.__len__()
    
    def setup_table(self, u_timeline_table):
        u_timeline_table.setHidden(True)
        u_timeline_table.setRowCount(self.get_data_count())

        for row_index in range(self.get_data_count()):
            u_timeline_table.setItem(row_index, 0, QTableWidgetItem(self.all_data.iat[row_index, 0]))
            u_timeline_table.setItem(row_index, 1, QTableWidgetItem(self.all_data.iat[row_index, 1][:-10]))
            u_timeline_table.setItem(row_index, 2, QTableWidgetItem(self.all_data.iat[row_index, 2][:-10]))
            try:
                if self.all_data.iat[row_index, 3] >= 0:
                    if self.all_data.iat[row_index, 3] == 1.0:
                        u_timeline_table.setItem(row_index, 3, QTableWidgetItem('Chewing'))
                    else:
                        u_timeline_table.setItem(row_index, 3, QTableWidgetItem('Non-chewing'))
                else:
                    u_timeline_table.setItem(row_index, 3, QTableWidgetItem('Not labeled yet'))
            except:
                pass
                # traceback.print_exc()
        u_timeline_table.setColumnCount(4)
        u_timeline_table.setHorizontalHeaderLabels(
            ('ID', 'start time', 'end time', 'labeling status')
        )
        u_timeline_table.setHidden(False)

    def get_navigate_to_arguments(self, row):
        p = self.all_data.iat[row, 0]
        time_start_str = self.all_data.iat[row, 1]
        time_end_str = self.all_data.iat[row, 2]
        return p, get_timestamp_from_activelearning_time(time_start_str), get_timestamp_from_activelearning_time(time_end_str)