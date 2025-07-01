#opt for CPD SR
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
import matplotlib.pyplot as plt
import copy
def read_csv_data(filename, encoding='utf-16'):
    potential = []
    current = []
    
    with open(filename, 'r', encoding=encoding) as file:
        lines = file.readlines()[6:]
        for line in lines:
            fields = line.strip().split(',')
            if len(fields) > 1:
                potential.append(float(fields[0]))
                current.append(float(fields[1]))
            else:
                continue
    return np.array(potential), np.array(current)


def derivative(x1, y1):
    return np.diff(y1) / np.diff(x1)


def get_algo_instance(model_search, model_cost, Alg_min_dis ,data):
    if model_search == 'Dynp':
        return rpt.Dynp(model=model_cost,min_size=Alg_min_dis, jump=1).fit(data)
    # elif model_search == 'Pelt':
    #     return rpt.Pelt(model=model_cost).fit(data)
    elif model_search == 'Binseg':
        return rpt.Binseg(model=model_cost,min_size=Alg_min_dis, jump=1).fit(data)
    elif model_search == 'BottomUp':
        return rpt.BottomUp(model=model_cost,min_size=Alg_min_dis, jump=1).fit(data)
    elif model_search == 'Window':
        return rpt.Window(width=40, model=model_cost,min_size=Alg_min_dis, jump=1).fit(data)
    else:
        raise ValueError(f"Unknown model search method: {model_search}")


def CPD( Alg_X, Alg_y, SM, CF, Alg_Thre_Factor, Alg_smooth_level = 2 ):

    Alg_Thre_Factor_min = 0.2
    window_factors = [50,20,3]
    filter_window_1 = max( int(len(Alg_y)/50),4) #in case data length is short
    if filter_window_1//2 ==0:
        filter_window_1 += 1
    smoothed = savgol_filter(Alg_y,  filter_window_1, 3)
    
    for i in range( Alg_smooth_level-1 ):

        filter_window = int(len(Alg_y)/window_factors[i+1])
        if filter_window//2 ==0:
            filter_window += 1
        smoothed_new = savgol_filter(smoothed,  filter_window,3)
        smoothed = copy.copy(smoothed_new)


    thre_region_len = Alg_Thre_Factor * len(Alg_X)
    dy_smooth = derivative(Alg_X, smoothed)  



    # Ruptures for smoothed data
    len_baseline_fitting_smooth = len(Alg_X)
    num_bkps_smooth = min(4,int(len(Alg_y)/10))
    result_smooth = []
    while len_baseline_fitting_smooth > thre_region_len  and num_bkps_smooth > 1:
        num_bkps_smooth -= 1
        # algo_smooth = rpt.Dynp(model=model_cost).fit(dy_smooth)


        algo_smooth = get_algo_instance(SM, CF, int(Alg_Thre_Factor_min * len_baseline_fitting_smooth), dy_smooth)

        result_smooth = algo_smooth.predict(n_bkps=num_bkps_smooth)
        len_baseline_fitting_smooth = max(result_smooth[:-1]) - min(result_smooth[:-1]) if result_smooth[:-1] else len(Alg_X)
        
    if  len_baseline_fitting_smooth < Alg_Thre_Factor_min * len_baseline_fitting_smooth:
        CP_info_boundary_smooth = (0,0)
    else:
        CP_info_boundary_smooth = ( int(max(result_smooth[:-1])), int(min(result_smooth[:-1]))) if result_smooth[:-1] else (0, 0)  #index
    CP_info_boundary_smooth_value = (Alg_X[CP_info_boundary_smooth[1]], Alg_X[CP_info_boundary_smooth[0]])   #value

    # np.save(file_name + '_CP_info_boundary_index_'+str(Alg_Thre_Factor )+'.npy', CP_info_boundary_smooth)
    # np.save(file_name + '_CP_info_boundary_value_'+str(Alg_Thre_Factor )+'.npy', CP_info_boundary_smooth_value)
    
    
    return (CP_info_boundary_smooth, CP_info_boundary_smooth_value,smoothed)


