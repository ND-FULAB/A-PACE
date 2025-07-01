#calculate the fitted baselines from different algorithms

import Change_Point_Detection 
#import pspython.pspyfiles as pspyfiles
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
import copy
from tqdm import tqdm
import warnings






def get_algo_instance(Alg, Alg_X, Alg_Data, Alg_Fit_Order, Alg_Num_Iter, Alg_Weight):
    #region polynomial 
    if Alg == 'goldindec':
        try:
            return ( (  pybaselines.polynomial.goldindec(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)

    elif Alg == 'imodpoly':
        try:
            return ( (  pybaselines.polynomial.imodpoly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'loess':
        try:
            return ( (  pybaselines.polynomial.loess(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)

    elif Alg == 'modpoly':
        try:
            return ( (  pybaselines.polynomial.modpoly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)

    elif Alg == 'penalized_poly':
        try:
            return ( (  pybaselines.polynomial.penalized_poly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)

    elif Alg == 'poly':
        try:
            return ( (  pybaselines.polynomial.poly(Alg_Data, poly_order=Alg_Fit_Order, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)


    elif Alg == 'quant_reg':
        try:
            return ( (  pybaselines.polynomial.quant_reg(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion
    # region spline
    elif Alg == 'corner_cutting':
        try:
            return ( (  pybaselines.spline.corner_cutting(Alg_Data, max_iter=Alg_Num_Iter) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'irsqr':
        try:
            return ( (  pybaselines.spline.irsqr(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'mixture_model':
        try:
            return ( (  pybaselines.spline.mixture_model(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_airpls':
        try:
            return ( (  pybaselines.spline.pspline_airpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_arpls':
        try:
            return ( (  pybaselines.spline.pspline_arpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_asls':
        try:
            return ( (  pybaselines.spline.pspline_asls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_aspls':
        try:
            return ( (  pybaselines.spline.pspline_aspls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_derpsalsa':
        try:
            return ( (  pybaselines.spline.pspline_derpsalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_drpls':
        try:
            return ( (  pybaselines.spline.pspline_drpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_iarpls':
        try:
            return ( (  pybaselines.spline.pspline_iarpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_iasls':
        try:
            return ( (  pybaselines.spline.pspline_iasls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_mpls':
        try:
            return ( (  pybaselines.spline.pspline_mpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_psalsa':
        try:
            return ( (  pybaselines.spline.pspline_psalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region whittaker
    elif Alg == 'airpls':
        try:
            return ( (  pybaselines.whittaker.airpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'arpls':
        try:
            return ( (  pybaselines.whittaker.arpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'asls':
        try:
            return ( (  pybaselines.whittaker.asls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'aspls':
        try:
            return ( (  pybaselines.whittaker.aspls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'derpsalsa':
        try:
            return ( (  pybaselines.whittaker.derpsalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'drpls':
        try:
            return ( (  pybaselines.whittaker.drpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'iarpls':
        try:
            return ( (  pybaselines.whittaker.iarpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'iasls':
        try:
            return ( (  pybaselines.whittaker.iasls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'psalsa':
        try:
            return ( (  pybaselines.whittaker.psalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region classification
    elif Alg == 'cwt_br':
        try:
            return ( (  pybaselines.classification.cwt_br(Alg_Data, poly_order=3, min_length=2, max_iter=Alg_Num_Iter, tol=0.01, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'dietrich':
        try:
            return ( (  pybaselines.classification.dietrich(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'fabc':
        try:
            return ( (  pybaselines.classification.fabc(Alg_Data, weights=Alg_Weight, weights_as_mask=True) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'fastchrom':
        try:
            return ( (  pybaselines.classification.fastchrom(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'golotvin':
        try:
            return ( (  pybaselines.classification.golotvin(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'rubberband':
        try:
            return ( (  pybaselines.classification.rubberband(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'std_distribution':
        try:
            return ( (  pybaselines.classification.std_distribution(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region misc
    elif Alg == 'beads':
        try:
            return ( (  pybaselines.misc.beads(Alg_Data, max_iter=Alg_Num_Iter) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'interp_pts':
        try:
            return ( (  pybaselines.misc.beads(Alg_Data, max_iter=Alg_Num_Iter) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion
    else:
        raise ValueError(f"Unknown Algorithm: {Alg}")
    

# def baseline_fitting_standard(  Alg_Boundarys, Alg_Raw_Current,Alg_Baseline_Current  ):
#     higher_counter = 0 #count the number of pints that baseline is higher than peak
#     index = Alg_Boundarys[1]
#     while index < Alg_Boundarys[0]+1:
#         if Alg_Baseline_Current[index] > Alg_Raw_Current[index]:
#             higher_counter += 1
#         index += 1
#     if higher_counter > (Alg_Boundarys[0]-Alg_Boundarys[1])*0.1:
#         return False
#     else:

#         len_compare = int( len(Alg_Raw_Current) - (Alg_Boundarys[0]-Alg_Boundarys[1]+1))
#         errors = np.zeros( shape=(  len_compare  ) )

#         for j in range(len_compare ):
#             if j < Alg_Boundarys[1]:
#                 errors[j] = abs(Alg_Raw_Current[j]-Alg_Baseline_Current [j])
#             else:
#                 errors[j] = abs(  Alg_Raw_Current[ j+(Alg_Boundarys[0]-Alg_Boundarys[1]+1)] -  Alg_Baseline_Current [j+(Alg_Boundarys[0]-Alg_Boundarys[1]+1) ])



#         sums = sum( Alg_Baseline_Current[:Alg_Boundarys[1]] ) + sum( Alg_Baseline_Current [Alg_Boundarys[0]:] )


#         if sum(errors) < sums*0.2:
#             # print(i,'in')
#            return True
#         else:

#             return False

# def extreme_baseline_detection(  Alg_Baselines,Alg_Current,Alg_CP_index  ):

#     Alg_Baselines= np.array(Alg_Baselines)
#     features = []
#     for curve in Alg_Baselines:
#         features.append([
#             np.mean(curve),
#             np.std(curve),
#             np.min(curve),
#             np.max(curve)
#         ])
#     features = np.array(features)

#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
#     clf_if = IsolationForest(random_state=42)
#     clf_if.fit(features_scaled)
#     outliers_if = clf_if.predict(features_scaled)
#     clf_lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')
#     outliers_lof = clf_lof.fit_predict(features_scaled)   

#     non_outliers_mask = (outliers_if == 1) & (outliers_lof == 1)
#     #print('normal curve:', non_outliers_mask)
#     non_outliers = Alg_Baselines[non_outliers_mask]
#     mean_values = np.mean(non_outliers, axis=0)
#     std_values = np.std(non_outliers, axis=0)
#     n_non_outliers = len(non_outliers)
#     confidence_level = 0.95
#     degrees_freedom = n_non_outliers - 1
#     confidence_interval = stats.t.interval(
#         confidence_level, 
#         degrees_freedom, 
#         loc=mean_values, 
#         scale=std_values / np.sqrt(n_non_outliers)
#     )
#     ci = []
#     for array in confidence_interval:
#         ci.append( array.tolist() )

#     return mean_values.tolist(), ci, non_outliers_mask
#     # else:
#     #     return [],[],[]


# def peak_info(  Alg_Potential, Alg_Current ):
#     peak_smooth = savgol_filter(Alg_Current, 5, 2)
#     peak_value = np.max(peak_smooth)
#     peak_index = np.argmax(peak_smooth) 
#     peak_location = Alg_Potential[   peak_index ]

#     return  peak_value, peak_location, peak_index


def cal_baselines(  args ):
    Alg_File_Name, Alg_Data,Fit_Alg = args  #data of file
    Num_Curves =  len( Alg_Data ) 
    Fit_Order = 3
    Num_Iter = 9999
    data_file = {}
    for i in range( 1):
        data_file['Curve No. '+str(i+1)] = { }
        Curve_CP_index = Alg_Data['Curve No. '+str(i+1)][ "Change Point Indexes "]
        #dropout
        if Curve_CP_index[0] == Curve_CP_index[1]:
            data_file['Curve No. '+str(i+1)]
            continue
        # Curve_CP_value = Alg_Data['Curve No. '+str(i+1)][ "Change Point Values "]

        Curve_Potential = Alg_Data['Curve No. '+str(i+1)]['Raw Poetntial ']
        Curve_Current = Alg_Data['Curve No. '+str(i+1)]['Raw Current']

        mask = np.ones(shape = len(Curve_Potential))
        mask[ int(Curve_CP_index[1]): int(Curve_CP_index[0])] = 0#be consist with boundary calculation 
        weight = mask.astype(bool) 

        for fitting_alg in Fit_Alg:

            (baseline, para), error = get_algo_instance(fitting_alg, Curve_Potential,Curve_Current,Fit_Order,Num_Iter,weight)
            if error:
                print(Alg_File_Name,fitting_alg,error)
            elif np.isnan(baseline[0]):  #avoid NaN in json file
                print(Alg_File_Name,fitting_alg,'is Nan'  )
            else:
                data_file['Curve No. '+str(i+1)][fitting_alg] = list(baseline)

    return (Alg_File_Name, data_file)



if __name__ == '__main__':
    # Mute all warnings

    CPD_alg_set = [
        ['Dynp','rank'],

        ['BottomUp','rank']


    ]
    threshold = 0.65

    for CPD_alg in CPD_alg_set :
        with open('data_new_'+ str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json', 'r', encoding='utf-8') as file:
            data_cpd = json.load(file)

        os.chdir('../')
        warnings.filterwarnings("ignore")
        root_loc = os.getcwd() 
        # Fit_Alg = ['airpls','fabc','goldindec','iarpls','imodpoly','iasls','modpoly',
        #     'pspline_airpls',  'pspline_asls', 'pspline_derpsalsa', 'pspline_drpls', 
        #     'pspline_mpls','penalized_poly', 'poly', 'quant_reg']

        fit_alg =  ['goldindec','imodpoly', 'modpoly', 'poly', 'quant_reg','penalized_poly', # polynomial (missing loess)
                    'irsqr', 'mixture_model' ,'pspline_airpls', 'pspline_arpls',  'pspline_aspls', 'pspline_derpsalsa', 'pspline_drpls', 'pspline_iarpls', 'pspline_mpls', 'pspline_psalsa',#spline (No wright: corner_cutting, weight problem: irqr ),'mixture_model'
                    'airpls','arpls','aspls','derpsalsa','drpls','iarpls','psalsa',#whittaker
                    'cwt_br','dietrich', 'fabc','fastchrom','golotvin','rubberband','std_distribution']#classification

        data_baselines = {}


        num_folders = len(data_cpd)


        num_cpus = cpu_count()
        #num_cpus = cpu_count()
        #num_cpus = 1
        # for i in range(len(raw_data)):
        #     print(i+1)
        #     args = (i, raw_data[i],num_curves[i] )
        #     process_file(args)


        for folder_index in range( num_folders ):
            
            # if folder_index == 6:
            #     break
            folder = str(folder_index + 1)
            print(folder)
            data_baselines[folder] = {}
            os.chdir( folder )

            folder_path =  os.path.join(root_loc,folder)
            file_list  = []

            #exclude path.txt
            for file_in_folder in os.listdir(folder_path):
                if file_in_folder .endswith('.pssession'):
                    file_list.append(file_in_folder )




            args = [ (file_list[i],data_cpd[folder][file_list[i]],fit_alg  ) for i in range(len(file_list)) ]


            print(f"Using {num_cpus} CPUs for Analysis" )

            with Pool(processes=num_cpus) as pool:
                #results = pool.map(process_file, args) 

                results = []
                # start_time = time.time()
                for result in tqdm(pool.imap(cal_baselines, args), total=len(args)):
                    results.append(result)
                    # elapsed_time = time.time() - start_time
                    # tqdm.write(f"Elapsed time: {elapsed_time:.2f} seconds")


            for result in results: #Alg_File_Index ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location
                file_name,baselines =  result

                data_baselines[folder][ file_name ] = baselines

            os.chdir(  root_loc )
        os.chdir('alg_selection_final_MSE')
        save_filename = 'baselines_new_' + str(threshold) +'_'+ CPD_alg[0]   +'_' + CPD_alg[1] + '.json'
            # save result as JSON file
        with open(save_filename, 'w') as json_file:
            json.dump(data_baselines, json_file,indent=4,  ensure_ascii=False)
                    


