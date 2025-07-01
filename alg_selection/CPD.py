import Change_Point_Detection
import pspython.pspyfiles as pspyfiles
from multiprocessing import Pool,cpu_count,freeze_support
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
import gc

from tqdm import tqdm



def read_pssession_file( filename):

    Measurements = pspyfiles.load_session_file( filename, load_peak_data=True, load_eis_fits=True)
    curves = [[],[]]

    len_curves = len(Measurements) 

    curves[0] =  [measurement.potential_arrays[0][20:-20]for measurement in Measurements]
    curves[1] =  [measurement.current_arrays[0][20:-20]for measurement in Measurements]
    times=  [measurement.timestamp for measurement in Measurements]

    return (curves,times,len_curves)


def process_file(args):
    # Alg_File_Index, Alg_Data, Num_Curves   = args 
    File, Alg_seaarch_model, Alg_cost_function,Alg_noise_level,Alg_threshold = args

    File_data_save= {}

    file_read_result =  read_pssession_file(File)

    a = file_read_result[2]#numger of curves

    for i in range(a):
        Curve_CP_index, Curve_CP_value,curve_smooth = Change_Point_Detection.CPD( file_read_result[0][0][i],file_read_result[0][1][i],Alg_seaarch_model, Alg_cost_function,Alg_threshold, Alg_noise_level )
        File_data_save['Curve No. '+str(i+1)] = {
                        'Date and time measurement': file_read_result[1][i],
                        'Raw Poetntial ': file_read_result[0][0][i],
                        'Raw Current': list(curve_smooth),
                        'Change Point Indexes ': Curve_CP_index,
                        'Change Point Values ' : Curve_CP_value
            }


    return  File,File_data_save





if __name__ == '__main__':


    # num_folder = os.listdir(root_loc)

    data_save = {}
    num_curves = []
    raw_data = []#save C,V info, curves in one file is one sub list

    curve_index = []
    num_cpus = cpu_count()
    #num_cpus = 12
    print(f"Using {num_cpus} CPUs for Analysis" )
    num_folders = 40

    threshold = 0.65
    noise_levels = [

        2,2,1,1,2,#5
        2,2,2,2,1,#10
        1,3,2,1,3,#15
        2,2,2,1,2,#20
        3,3,2,3,2,#25
        3,2,2,1,2,#30
        1,3,3,1,1,#35
        1,1,1,2,1 #40
    ]


    CPD_alg_set = [
        ['Dynp','rank'],

        ['BottomUp','rank'],


    ]


    for CPD_alg in CPD_alg_set :
        os.chdir('../')
        root_loc = os.getcwd() 
        for folder_index in range( num_folders ):
            folder = str(folder_index + 1)
            print(folder)
            # if folder == '11':
            #     break
            data_save[folder] = {}
            os.chdir( folder )
            folder_path =  os.path.join(root_loc,folder)
            file_list  = []

            #exclude path.txt
            for file_in_folder in os.listdir(folder_path):
                if file_in_folder .endswith('.pssession'):
                    file_list.append(file_in_folder )

            
            # for file in os.listdir( folder_path ):


            args = [(file_list[i], CPD_alg[0], CPD_alg[1], noise_levels[folder_index],threshold) for i in range(len(file_list))]


            with Pool(processes=num_cpus) as pool:
                #results = pool.map(process_file, args) 

                results = []
                # start_time = time.time()
                for result in tqdm(pool.imap(process_file, args), total=len(args)):
                    results.append(result)
                    # elapsed_time = time.time() - start_time
                    # tqdm.write(f"Elapsed time: {elapsed_time:.2f} seconds")

                    for result in results: #Alg_File_Index ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location
                        file,Data =  result
                        # file_name_save = file_name[file_index]
                        for file_index in range(len(file_list)):
                            data_save[folder][file] = Data

            os.chdir(  root_loc )
        #'data_new.json' for small filter window in CPD 
        os.chdir('alg_selection_final_MSE')
        save_filename = 'data_new_'+ str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json'
            # save result as JSON file
        with open(save_filename, 'w') as json_file:
            json.dump(data_save, json_file,indent=4,  ensure_ascii=False)
