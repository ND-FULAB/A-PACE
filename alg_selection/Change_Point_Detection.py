#opt for CPD SR
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
import matplotlib.pyplot as plt
import copy


def valid_savgol_window(sample_count, requested_window, polyorder=3):
    if sample_count < 1:
        raise ValueError("cannot smooth an empty signal")
    max_window = sample_count if sample_count % 2 else sample_count - 1
    min_window = polyorder + 1
    if min_window % 2 == 0:
        min_window += 1
    if max_window < min_window:
        raise ValueError(
            f"signal requires at least {min_window} samples for polyorder {polyorder}"
        )
    window = max(int(requested_window), min_window)
    if window % 2 == 0:
        window += 1
    return min(window, max_window)


def smooth_signal(signal, smooth_level=2, polyorder=3):
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains non-finite values")
    if isinstance(smooth_level, bool) or not isinstance(
        smooth_level, (int, np.integer)
    ) or smooth_level not in (1, 2, 3):
        raise ValueError("smooth level must be an integer from 1 to 3")
    smoothed = values
    for factor in (50, 20, 3)[:smooth_level]:
        window = valid_savgol_window(len(values), len(values) / factor, polyorder)
        smoothed = savgol_filter(smoothed, window_length=window, polyorder=polyorder)
    return smoothed


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
    Alg_X = np.asarray(Alg_X, dtype=float)
    Alg_y = np.asarray(Alg_y, dtype=float)
    if Alg_X.ndim != 1 or Alg_y.ndim != 1:
        raise ValueError("potential and current must be one-dimensional")
    if len(Alg_X) != len(Alg_y):
        raise ValueError("potential and current must have the same length")
    if len(Alg_X) < 5:
        raise ValueError("at least 5 samples are required for change-point detection")
    if not np.all(np.isfinite(Alg_X)) or not np.all(np.isfinite(Alg_y)):
        raise ValueError("potential and current must contain only finite values")
    if np.any(np.diff(Alg_X) == 0):
        raise ValueError("potential values must not contain adjacent duplicates")
    try:
        Alg_Thre_Factor = float(Alg_Thre_Factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("peak-region threshold must be numeric") from exc
    if not np.isfinite(Alg_Thre_Factor) or not 0 < Alg_Thre_Factor <= 1:
        raise ValueError("peak-region threshold must be in (0, 1]")

    Alg_Thre_Factor_min = 0.2
    smoothed = smooth_signal(Alg_y, Alg_smooth_level, polyorder=3)


    thre_region_len = Alg_Thre_Factor * len(Alg_X)
    dy_smooth = derivative(Alg_X, smoothed)  



    # Ruptures for smoothed data
    len_baseline_fitting_smooth = len(Alg_X)
    num_bkps_smooth = min(4,int(len(Alg_y)/10))
    result_smooth = []
    while len_baseline_fitting_smooth > thre_region_len  and num_bkps_smooth > 1:
        num_bkps_smooth -= 1
        # algo_smooth = rpt.Dynp(model=model_cost).fit(dy_smooth)


        min_size = max(1, int(Alg_Thre_Factor_min * len(Alg_X)))
        algo_smooth = get_algo_instance(SM, CF, min_size, dy_smooth)

        result_smooth = algo_smooth.predict(n_bkps=num_bkps_smooth)
        len_baseline_fitting_smooth = max(result_smooth[:-1]) - min(result_smooth[:-1]) if result_smooth[:-1] else len(Alg_X)
        
    if len_baseline_fitting_smooth < Alg_Thre_Factor_min * len(Alg_X):
        CP_info_boundary_smooth = (0,0)
    else:
        CP_info_boundary_smooth = ( int(max(result_smooth[:-1])), int(min(result_smooth[:-1]))) if result_smooth[:-1] else (0, 0)  #index
    CP_info_boundary_smooth_value = (Alg_X[CP_info_boundary_smooth[1]], Alg_X[CP_info_boundary_smooth[0]])   #value

    # np.save(file_name + '_CP_info_boundary_index_'+str(Alg_Thre_Factor )+'.npy', CP_info_boundary_smooth)
    # np.save(file_name + '_CP_info_boundary_value_'+str(Alg_Thre_Factor )+'.npy', CP_info_boundary_smooth_value)
    
    
    return (CP_info_boundary_smooth, CP_info_boundary_smooth_value,smoothed)


