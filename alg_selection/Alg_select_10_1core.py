#23793 files

import json
from itertools import combinations
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
import pybaselines 
import numpy as np
from datetime import datetime
import os
import re
import json
import gc
#import pandas as pd
import math
from tqdm import tqdm
from scipy.stats import pearsonr
import sys


def baseline_fitting_standard(  Alg_Boundarys, Alg_Raw_Current,Alg_Baseline_Current, Alg_File_Name_error ):
    

    Alg_Raw_Current = np.array(Alg_Raw_Current)
    Alg_Baseline_Current = np.array(Alg_Baseline_Current)
    higher_counter = np.sum(Alg_Baseline_Current[Alg_Boundarys[1]:Alg_Boundarys[0]+1] > [ a*1.00 for a in Alg_Raw_Current[Alg_Boundarys[1]:Alg_Boundarys[0]+1]])
    if higher_counter > (Alg_Boundarys[0] - Alg_Boundarys[1]) * 0.1:
        return False
    
    Alg_slope = (Alg_Raw_Current[Alg_Boundarys[1]] - Alg_Raw_Current[Alg_Boundarys[0]]  )/(  Alg_Boundarys[1] - Alg_Boundarys[0] )
    Alg_cons =  Alg_Raw_Current[Alg_Boundarys[1]] - Alg_slope * Alg_Boundarys[1]
    Alg_area_linear = 0
    Alg_area_baseline = 0
    # points4compare = []
    for point_index in range( Alg_Boundarys[1] , Alg_Boundarys[0]  ):
        diff = Alg_Raw_Current[point_index] -  (Alg_slope * point_index + Alg_cons)
        if diff > 0 :
            Alg_area_linear += diff
            Alg_area_baseline += abs( Alg_Raw_Current[point_index] - Alg_Baseline_Current[point_index]  )
    
    if Alg_area_linear > 0:
        if  ( Alg_area_baseline - Alg_area_linear  )/Alg_area_linear < -0.3:
            print('Alg_File_Name_error Area: ', ( Alg_area_baseline - Alg_area_linear  )/Alg_area_linear  )
            return False
    # else:
    #     print('overpeak')


    # else:
        # errors = np.abs(np.concatenate((Alg_Raw_Current[:Alg_Boundarys[1]] - Alg_Baseline_Current[:Alg_Boundarys[1]], Alg_Raw_Current[Alg_Boundarys[0]+1:] - Alg_Baseline_Current[Alg_Boundarys[0]+1:])))
        # sums = np.sum(Alg_Baseline_Current[:Alg_Boundarys[1]]) + np.sum(Alg_Baseline_Current[Alg_Boundarys[0]+1:])
    Alg_raw = np.concatenate(( Alg_Raw_Current[ :Alg_Boundarys[1] ], Alg_Raw_Current[Alg_Boundarys[0]: ]))
    Alg_baseline = np.concatenate(( Alg_Baseline_Current[ :Alg_Boundarys[1] ], Alg_Baseline_Current[Alg_Boundarys[0]: ]))
    mid_point  = len(  Alg_Raw_Current[ :Alg_Boundarys[1] ]) 
    sigma_2 = ( 0.25*(len(Alg_raw)) ) **2 
    weights = []
    Square_Error = [] # (y-y*)**2
    for point_index in range( len(Alg_raw)  ):
        if point_index < mid_point :
            Square_Error.append( (Alg_raw[point_index] - Alg_baseline[point_index]) **2  )
            weights.append( math.exp( - ( point_index  - mid_point-1)**2/sigma_2   )   )
        else:
            Square_Error.append( (Alg_raw[point_index] - Alg_baseline[point_index] )**2  )
            weights.append( math.exp( - ( point_index  - (mid_point) )**2/sigma_2   )   )
    sum_weights = sum(weights)
    
    MWSE =  0 

    for point_index in range( len(Alg_raw)  ):
        MWSE += weights[point_index] * Square_Error[point_index]
    
    MWSE = MWSE/sum_weights/( np.max(Alg_raw) - np.min(Alg_raw)**2   )
    if MWSE > 0.1:
        print('MWSE: ', MWSE)
    #print(pearson_r,p_value)
    return MWSE < 0.1


def extreme_baseline_detection(  Alg_Baselines,Alg_Current,Alg_CP_index  ):

    Alg_Baselines= np.array(Alg_Baselines)
    features = []
    for curve in Alg_Baselines:
        features.append([
            np.mean(curve),
            np.std(curve),
            np.min(curve),
            np.max(curve)
        ])
    features = np.array(features)

    scaler = StandardScaler()
    try:
        features_scaled = scaler.fit_transform(features)
    except:
            print('BL',np.shape(features))
            print('FT',np.shape(Alg_Baselines))
    clf_if = IsolationForest(random_state=42)
    clf_if.fit(features_scaled)
    outliers_if = clf_if.predict(features_scaled)
    clf_lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')
    outliers_lof = clf_lof.fit_predict(features_scaled)   

    # non_outliers_mask = (outliers_if == 1) & (outliers_lof == 1)
    non_outliers_mask = outliers_lof == 1
    #print('normal curve:', non_outliers_mask)
    non_outliers = Alg_Baselines[non_outliers_mask]

    median_values = np.median(non_outliers, axis=0)
    q1 = np.percentile(non_outliers, 25, axis=0)
    q3 = np.percentile(non_outliers, 75, axis=0)
    iqr_values = q3 - q1
    n_non_outliers = len(non_outliers)

    # 估计标准误差
    se_median = (1.253 * iqr_values) / np.sqrt(n_non_outliers)

    # 设置置信水平
    confidence_level = 0.99
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # 计算置信区间
    ci_lower = median_values - z_score * se_median
    ci_upper = median_values + z_score * se_median
    ci = [ci_lower.tolist(), ci_upper.tolist()]

    return median_values.tolist(), ci, non_outliers_mask
    # else:
    #     return [],[],[]


def get_CI(  Alg_Baselines):
    Alg_Baselines= np.array(Alg_Baselines)
    median_values = np.median(Alg_Baselines, axis=0)
    q1 = np.percentile(Alg_Baselines, 25, axis=0)
    q3 = np.percentile(Alg_Baselines, 75, axis=0)
    iqr = q3 - q1
    n_samples = Alg_Baselines.shape[0]
    se_median = (1.253 * iqr) / np.sqrt(n_samples)
    confidence_level = 0.99
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    ci_lower = median_values - z_score * se_median
    ci_upper = median_values + z_score * se_median
    ci = [ci_lower.tolist(), ci_upper.tolist()]
    return median_values.tolist(), ci


def peak_info(  Alg_Potential, Alg_Current ):
    # peak_smooth = savgol_filter(Alg_Current, 5, 2)
    peak_index = np.argmax( Alg_Current ) 
    peak_location = Alg_Potential[   peak_index ]

    return   Alg_Current[peak_index], peak_location, peak_index


def alg_select( args ):
    Alg_File_Name, Alg_Data_Baseline,Alg_Data_CPD, Algs = args  #data of file
    Baseline_Mean = [] 
    Peak_Mean = []
    Peak_Index = []
    baselines_select = [] #collect baselines
    Num_Curves =  len( Alg_Data_Baseline )
    for curve_index in range( Num_Curves ):
        # consisit with the return result(when Curve_CP_index[0] == Curve_CP_index[1] ) in Alg_cal.py cal_baseline() function 
        if not  Alg_Data_Baseline['Curve No. '+str(curve_index+1)]:
            Baseline_Mean.append([])

            Peak_Mean.append(0) 
            Peak_Index.append(0)
            continue
        
        Curve_CP_index = Alg_Data_CPD['Curve No. '+str(curve_index+1)][ "Change Point Indexes "]
        Curve_Potential = Alg_Data_CPD['Curve No. '+str(curve_index+1)]['Raw Poetntial ']
        Curve_Current = Alg_Data_CPD['Curve No. '+str(curve_index+1)]['Raw Current']
                    
        if Curve_CP_index[0] == Curve_CP_index[1]:
            print( 'wrong', Alg_File_Name,'Curve No. '+str(curve_index+1), 'Index: ',int(Curve_CP_index[1]),int(Curve_CP_index[0])) 
        
        for alg in Algs :
            
            if alg  not in Alg_Data_Baseline['Curve No. '+str(curve_index+1)]: # this alg fails in this curve
                continue
            alg2baseline =  Alg_Data_Baseline['Curve No. '+str(curve_index+1)][alg] # baseline from single alg in one alg set
            if baseline_fitting_standard(  Curve_CP_index,Curve_Current,alg2baseline,Alg_File_Name ):
                baselines_select.append( alg2baseline)
        #deopout
        if len(baselines_select) == 0:#not enough data for extreme data detection , drop 
            Baseline_Mean.append([])
            Peak_Mean.append(0) 
            Peak_Index.append(0)
            continue
        # if len(baselines_select) < 5:#not enough data for extreme data detection , drop 
        #     Baseline_Mean.append([])
        #     Peak_Mean.append(0) 
        #     Peak_Index.append(0)
        #     continue
            
        # Curve_Baseline_Mean,Curve_Outlier_Info =  extreme_baseline_detection(baselines_select,Curve_Current,Curve_CP_index)
        # Baseline_Mean.append(Curve_Baseline_Mean)

        if len(baselines_select) <5:  #no extreme value detection
            Curve_Baseline_Mean, Curve_Baseline_CI =get_CI(baselines_select)
            Baseline_Mean.append(Curve_Baseline_Mean)

        else:
            Curve_Baseline_Mean, Curve_Baseline_CI,Curve_Outlier_Info =  extreme_baseline_detection(baselines_select,Curve_Current,Curve_CP_index)
            Baseline_Mean.append(Curve_Baseline_Mean)


        if Curve_Baseline_Mean:
           
            Curve_Peak_Mean,Curve_Peak_Location,Curve_Peak_Index = peak_info(Curve_Potential[ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a - b for a, b in zip(Curve_Current, Curve_Baseline_Mean)][ int(Curve_CP_index[1]):int(Curve_CP_index[0])]  )
            Peak_Mean.append(Curve_Peak_Mean)
            Peak_Index.append(Curve_Peak_Index)
            
        else:#dropout
            # print(Alg_File_Name, curve_index,'Failed')
            Baseline_Mean.append([]) 
            Peak_Mean.append(0) 
            Peak_Index.append(0)
        #print(Alg_File_Name, Baseline_Mean, Peak_Mean,Peak_Location)
    return Alg_File_Name, Baseline_Mean, Peak_Mean,Peak_Index
            


if __name__ == '__main__':

    threshold = 0.65
    CPD_alg_set = [
        ['Dynp','rank'],

        ['BottomUp','rank']


    ]


    fit_alg_all = [

        [
'quant_reg',
'derpsalsa',
'pspline_airpls',
'pspline_mpls',
'pspline_derpsalsa',
'dietrich',
'pspline_psalsa',
'psalsa',
'std_distribution',
'fastchrom',
        ],
        [
'quant_reg',
'pspline_derpsalsa',
'derpsalsa',
'imodpoly',
'penalized_poly',
'goldindec',
'modpoly',
'dietrich',
'std_distribution',
'fastchrom',
        ]

    ]



    with open('folder_file_list.json', 'r', encoding='utf-8') as file:
        data_folderlists = json.load(file)

    CPD_alg_index = 0
    for CPD_alg in CPD_alg_set :


        file_path_baseline = 'baselines_new_' + str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json'
        file_path_CPD = 'data_new_'+ str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json'

        root_loc = os.getcwd() 
        with open(file_path_baseline, 'r', encoding='utf-8') as file:
            data_baselines = json.load(file)
        with open(file_path_CPD, 'r', encoding='utf-8') as file:
            data_CPD = json.load(file)

        # print(len(data_baselines))


        fit_alg = fit_alg_all[CPD_alg_index]


        num_alg_min = 1 # number in set - number fixed
        num_alg_max =  len(fit_alg)

        alg_sets = []

        for r in range(num_alg_min, num_alg_max+1):

            for comb in combinations(fit_alg, r):
                alg_sets.append( comb)
        #print(alg_sets)
        print('Totally Alg combination number: ', len(alg_sets) )
        num_folders = len( data_baselines)


        num_cpus = cpu_count()
        num_cpus = 1
        print(f"Using {num_cpus} CPUs for Analysis" )
    #alg set - structure folder - file - curve

        # scores=[[],[]] #0:drop rate, 1:  STD diff
        peak_info2date = {}

        for single_alg_set in alg_sets:
            peak_info2date[str(single_alg_set)]={}
            # peaks = []
            total_num_curves = 0
            total_num_drop = 0
            
            

            now = datetime.now()

            
            for folder_index in range( num_folders):
                
                net_peaks = []  #save net peak values 
                raw_peaks = []  # save peak values in raw data
                folder = str(folder_index + 1)
                peak_info2date[str(single_alg_set)][folder] = {}

                # os.chdir( folder )

                # folder_path =  os.path.join(root_loc,folder)
                # file_list  = []

                
                data_baselines_folder = data_baselines[folder]
                data_CPD_folder = data_CPD[folder]
                        
                file_list = data_folderlists[folder]
                # for single_alg_set in alg_sets:
                # # print(folder)    
                # args = [ ( file_list[i], data_baselines_folder[file_list[i]], data_CPD_folder[file_list[i]],single_alg_set ) for i in range(len(file_list)) ]



                # with Pool(processes=num_cpus) as pool:
                #     results = pool.map(alg_select, args) 
                results = []
                for i in range(len(file_list)):
                    results.append(alg_select( ( file_list[i], data_baselines_folder[file_list[i]], data_CPD_folder[file_list[i]],single_alg_set ))  )


                peak_folder = []
                raw_peak_folder = []
                # num_curves_folder = 0
                # num_drop_folder = 0
                for result in results: # reuslt:file level ; results:folder level
                    file_name,  baseline_mean, peak_mean, peak_index=  result
                    num_curves_file = len(baseline_mean)
                    total_num_curves += num_curves_file
                    peak_file = []
                    raw_peak_file = []
                    # num_dropout = 0
                    average_raw_peak = -1
                    average_peak = -1

                    for curve_index in range(num_curves_file): #curve level
                        if not baseline_mean[curve_index]:
                            total_num_drop += 1
                            continue
                        peak_file.append ( peak_mean[curve_index]  )
                        raw_peak_file.append(  data_CPD_folder[file_name]['Curve No. ' + str(curve_index+1)]['Raw Current'][peak_index[curve_index]]  )

                    if len(raw_peak_file) != 0:
                        average_raw_peak = sum( raw_peak_file)/len(raw_peak_file)

                    if len(peak_file) != 0:
                        average_peak = sum( peak_file)/len(peak_file)
                    peak_folder.append( average_peak )
                    raw_peak_folder.append(average_raw_peak)
                    # num_drop_folder.append( num_dropout )



                peak_info2date[str(single_alg_set)][folder]={
                    # 'Number of Curves ' : num_curves_folder,
                    # 'Number of Dropout ' : num_drop_folder,
                    'Raw Data Peak ': raw_peak_folder ,
                    'Net Peak ':peak_folder 
                }

                # os.chdir(  root_loc )
            

            peak_info2date[str(single_alg_set)]['Algorithms: '] =list(single_alg_set)
            peak_info2date[str(single_alg_set)]['Total Number of Curves'] = total_num_curves
            peak_info2date[str(single_alg_set)]['Total Number of Fails'] = total_num_drop
            print('Time Spend: ',(datetime.now()-now)   )   
        

        save_filename =  'result_peak_info_result_50_'+ str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json'
            # save result as JSON file
        with open(save_filename, 'w') as json_file:
            json.dump(peak_info2date, json_file,indent=4,  ensure_ascii=False)
        CPD_alg_index += 1
                    


