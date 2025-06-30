#rerun CPD after modifying CPs
import pspython.pspyfiles as pspyfiles
import Change_Point_Detection
from multiprocessing import Pool,cpu_count,freeze_support
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Algs import get_algo_instance
import numpy as np
import re
import datetime 
import os
from dateutil import parser
import json
import gc
import pandas as pd
import copy
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
from scipy.signal import savgol_filter
import shutil
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from demo import baseline_fitting_standard,extreme_baseline_detection,get_CI,peak_info
import ast
def process_file(args):
    Alg_File_Name,Alg_Curve_index,Curve_CP_value, Fit_Alg ,Alg_noise_level   = args 
    with open('database/results.json', 'r', encoding='utf-8') as file:
        data_result = json.load(file)
    Alg_current = data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Raw Current"]
    Alg_potential = data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Raw Poetntial "]
    # print(Boundary)
    # x, y = read_csv_data(File_Name)
    Curve_CP_index = []

    Baseline_Mean = [] 
    Baseline_CI = []
    Peak_Mean = []
    Peak_Max = []
    Peak_Min = []
    Peak_Location = []
    # for i in range( Num_Curves ):
        # print(Alg_File_Name,i,len(Alg_potential),len(Alg_current))

    window_factors = [50,20,3]
    filter_window_1 = max( int(len(Alg_current)/50),2) #in case data length is short
    if filter_window_1//2 ==0:
        filter_window_1 += 1
    smoothed = savgol_filter(Alg_current,  filter_window_1, 3)
    for i in range( Alg_noise_level-1 ):

        filter_window = int(len(Alg_current)/window_factors[i+1])
        if filter_window//2 ==0:
            filter_window += 1
        smoothed_new = savgol_filter(smoothed,  filter_window,3)
        smoothed = copy.copy(smoothed_new)


    Curve_CP_index.append(min(range(len(Alg_potential)), key=lambda i: abs(Alg_potential[i] - Curve_CP_value[1])))
    Curve_CP_index.append(min(range(len(Alg_potential)), key=lambda i: abs(Alg_potential[i] - Curve_CP_value[0])))

    # #dropout 
    # if Curve_CP_index[0] == Curve_CP_index[1]:
    #     print( Curve_CP_value, Curve_CP_index )
    #     Baseline_Mean.append([])
    #     Baseline_CI.append([])
    #     Peak_Mean.append(0) 
    #     Peak_Max.append(0) 
    #     Peak_Min.append(0) 
    #     Peak_Location.append(0)

    #     plt.figure(num = i+ 100)
    #     plt.rcParams['font.family'] = 'Arial'
    #     plt.rcParams['font.size'] = 14
    #     plt.figure(figsize=(16, 9))
    #     plt.plot(Alg_potential,Alg_current, label='Raw_data', color='red')
    #     plt.plot(Alg_potential,smoothed, label='CPD smooth', color='black')
    #     plt.axvline(x=Curve_CP_value[1], color='black', label='Boundary-peak' )
    #     plt.axvline(x=Curve_CP_value[0],color='black')
    #     plt.xlabel('Potential', fontsize=14, fontname='Arial')
    #     plt.ylabel('Current', fontsize=14, fontname='Arial')
    #     plt.legend()
    #     plt.title('Data Analysis Results(Change point overlap)')
    #     plt.savefig('Fig_Saved/'+Alg_File_Name + '_'+str(Alg_Curve_index) + 'alg.png')
    #     plt.close('all')
    #     continue

    Fit_Order = 5
    Num_Iter = 9999
    mask = np.ones(shape = len(Alg_potential))
    mask[ int(Curve_CP_index[1]): int(Curve_CP_index[0])] = 0#be consist with boundary calculation 
    weight = mask.astype(bool) 

    #Fit_Alg = ['imodpoly4', 'penalized_poly4', 'pspline_derpsalsa', 'pspline_iarpls', 'pspline_iasls', 'pspline_mpls', 'fabc']

    Alg_baselines = []
    Fit_Alg_using = copy.copy(Fit_Alg)  #this list is to store the using algs
    # plt.figure(num =   i+ 10000)
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 14
    # plt.figure(figsize=(16, 9))
    # plt.plot(Alg_potential,Alg_current, label='Raw_data', color='red')
    # plt.plot(Alg_potential,smoothed, label='CPD smooth', color='black')
    # plt.axvline(x=Curve_CP_value[1], color='black', label='Boundary-peak' )
    # plt.axvline(x=Curve_CP_value[0],color='black')
    index_baseline_fitting_standard = [] #index for baseline not satisfiled baseline_fitting_standard()        
    for fitting_alg in Fit_Alg:
        (baseline, para), error = get_algo_instance(fitting_alg,Alg_potential,smoothed,Fit_Order,Num_Iter,weight)
        if error:
            print(Alg_File_Name,Fit_Alg,error)

        elif baseline_fitting_standard(  Curve_CP_index, smoothed,baseline  ):
            # print(fitting_alg,'works')
            Alg_baselines.append(baseline)
            # plt.plot(Alg_potential,baseline, label=fitting_alg)
        else:
            Fit_Alg_using.remove( fitting_alg  )
            #plt.plot(Alg_potential,baseline, '--', label=fitting_alg)
            
    if len(Alg_baselines) == 0:
        Baseline_Mean.append([])
        Baseline_CI.append([])
        Peak_Mean.append(0) 
        Peak_Max.append(0) 
        Peak_Min.append(0) 
        Peak_Location.append(0)
        # plt.xlabel('Potential', fontsize=14, fontname='Arial')
        # plt.ylabel('Current', fontsize=14, fontname='Arial')
        # plt.legend()
        # plt.title('Data Analysis Results(Fitting Failed)')
        # plt.savefig('Fig_Saved/'+ Alg_File_Name + '_'+str(Alg_Curve_index) + 'alg.png')
        # plt.close('all')
        print('Can not fitting baseline')
    else:
        if len(Alg_baselines) <5:  #no extreme value detection
            Curve_Baseline_Mean, Curve_Baseline_CI =get_CI(Alg_baselines)
            Baseline_Mean.append(Curve_Baseline_Mean)
            Baseline_CI.append( Curve_Baseline_CI )
        else:
            Curve_Baseline_Mean, Curve_Baseline_CI,Curve_Outlier_Info =  extreme_baseline_detection(Alg_baselines)
            Baseline_Mean.append(Curve_Baseline_Mean)
            Baseline_CI.append( Curve_Baseline_CI )
        # for j in range(len(Fit_Alg_using) ):
        #     if Curve_Outlier_Info[j]:
        #         plt.plot(Alg_potential,Alg_baselines[j], label=Fit_Alg_using[j])
        #     else:
        #         plt.plot(Alg_potential,Alg_baselines[j], '-.' , label=Fit_Alg_using[j])


        if Curve_Baseline_Mean:

            Curve_Peak_Mean,Curve_Peak_Location,Curve_Peak_Index = peak_info( Alg_potential[ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a - b for a, b in zip(smoothed, Curve_Baseline_Mean)][ int(Curve_CP_index[1]):int(Curve_CP_index[0])]  )
            Peak_Mean.append(Curve_Peak_Mean)
            Peak_Location.append(Curve_Peak_Location)
            
            Peak_Min.append(Curve_Peak_Mean - Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ] )
            Peak_Max.append(Curve_Peak_Mean +  Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ])
            # plt.plot(Alg_potential,Curve_Baseline_Mean, label='baseline', color='yellow')
            # plt.fill_between(
            #     Alg_potential,
            #     [a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
            #     [a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
            #     color='blue', alpha=0.2, label='95% Confidence Interval of Baseline'
            # )
            # plt.xlabel('Potential', fontsize=14, fontname='Arial')
            # plt.ylabel('Current', fontsize=14, fontname='Arial')
            # plt.legend()
            # plt.title('Data Analysis Results')
            # plt.savefig('Fig_Saved/'+ Alg_File_Name  + '_'+str(Alg_Curve_index) + 'alg.png')
            # plt.close('all')



            # plt.figure(num = i+ 1)
            # plt.rcParams['font.family'] = 'Arial'
            # plt.rcParams['font.size'] = 14
            # plt.figure(figsize=(16, 9))
            # plt.plot(Alg_potential,Alg_current, label='Raw_data', color='red')
            # plt.plot(Alg_potential, Curve_Baseline_Mean, label='Baseline', color='blue')
            # plt.axvline(x=Curve_CP_value[1], color='red', label='Boundary-peak' )
            # plt.axvline(x=Curve_CP_value[0],color='red')
            # plt.fill_between(
            #     Alg_potential,
            #     [a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
            #     [a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
            #     color='blue', alpha=0.2, label='95% Confidence Interval of Baseline'
            # )
            # plt.plot(Alg_potential[ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a-b for a,b in zip(Alg_current,Curve_Baseline_Mean ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ], label='Peak', color='green')
            # # plt.fill_between(
            # #     Alg_potential[ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            # #     [a-b for a,b in zip(Alg_current,Curve_Baseline_CI[0] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            # #     [a-b for a,b in zip(Alg_current,Curve_Baseline_CI[1] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            # #     color='green', alpha=0.2, label='95% Confidence Interval of Peak'
            # # )
            # plt.scatter( Curve_Peak_Location,Curve_Peak_Mean,label= 'Peak Height' )
            # plt.xlabel('Potential', fontsize=14, fontname='Arial')
            # plt.ylabel('Current', fontsize=14, fontname='Arial')
            # plt.legend()
            # plt.title('Data Analysis Results')
            # plt.savefig('Fig_Saved/'+ Alg_File_Name + '_'+str(Alg_Curve_index) + '.png')

            # plt.figure(num =  Alg_File_Index + i+ 100) # back to draw alg figures


        else:
            print(Alg_File_Name, i,'Failed')
            Peak_Mean.append(0) 
            Peak_Max.append(0) 
            Peak_Min.append(0) 
            Peak_Location.append(0)
            # plt.xlabel('Potential', fontsize=14, fontname='Arial')
            # plt.ylabel('Current', fontsize=14, fontname='Arial')# back to draw alg figures
            # plt.legend()
            # plt.title('Data Analysis Results')
            # plt.savefig('Fig_Saved/'+ Alg_File_Name + '_'+str(Alg_Curve_index) + 'alg.png')
            # plt.close('all')
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Change Point Indexes "] = Curve_CP_index
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Change Point Values "] = Curve_CP_value
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Baseline Mean "] =  Curve_Baseline_Mean
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["95\\% Confidence Interval of Baseline: "] = Curve_Baseline_CI 
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Peak Value "] = Curve_Peak_Mean
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["95\\% Confidence Interval of Peak Value"] =[Curve_Peak_Mean - Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ] , Curve_Peak_Mean +  Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ]]
        data_result[Alg_File_Name]["Curve No. "+str(1+Alg_Curve_index)]["Peak Location: "] = Curve_Peak_Location
        save_filename = 'database/results.json'
        # save result as JSON file
        with open(save_filename, 'w') as json_file:
            json.dump(data_result, json_file,indent=4,  ensure_ascii=False)

    
SR_weight = 0.5

# Delete figures update
# with open('Algorithm Setting.json', 'r', encoding='utf-8') as file:
#     data_algs = json.load(file)
# name = str(int(SR_weight/0.01))
# A = ("100hz.pssession",0,[-0.5,-0.1], 
#     list(ast.literal_eval(data_algs[name][ "Baseline Fitting Algorithms"]) ),
#     3) 
# process_file(A)
    

