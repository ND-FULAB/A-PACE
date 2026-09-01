#opt for CPD SR
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
import matplotlib.pyplot as plt
import copy


def valid_savgol_window(sample_count, requested_window, polyorder=3):
    """Return a valid odd Savitzky-Golay window for the given data size."""
    if sample_count < 1:
        raise ValueError("cannot smooth an empty signal")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")

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
    """Apply the configured one-to-three Savitzky-Golay smoothing passes."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains non-finite values")
    if isinstance(smooth_level, bool) or not isinstance(
        smooth_level, (int, np.integer)
    ):
        raise ValueError("smooth level must be an integer from 1 to 3")
    if smooth_level not in (1, 2, 3):
        raise ValueError("smooth level must be from 1 to 3")

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


def get_algo_instance(model_search, model_cost, data):
    if model_search == 'Dynp':
        return rpt.Dynp(model=model_cost).fit(data)
    # elif model_search == 'Pelt':
    #     return rpt.Pelt(model=model_cost).fit(data)
    elif model_search == 'Binseg':
        return rpt.Binseg(model=model_cost).fit(data)
    elif model_search == 'BottomUp':
        return rpt.BottomUp(model=model_cost).fit(data)
    elif model_search == 'Window':
        return rpt.Window(width=40, model=model_cost).fit(data)
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




    # noise_level = np.std(Alg_y)

    
    # filer_win_factor = [25,10,3 ]
    # filer_win_index = 0
    # filter_window = int(len(Alg_y)/50)
    # if filter_window//2 ==0:
    #     filter_window += 1
    # #print('window size',filter_window)
    # smoothed = savgol_filter(Alg_y, window_length=filter_window , polyorder=3)
    # for i in range(len(filer_win_factor) ):  # 遍历奇数窗口长度
    #     residual_noise = np.std(smoothed)
    #     if residual_noise < 0.01:  # 调整这个阈值
    #         break

    #     filter_window = int(len(Alg_y)/int(filer_win_factor[filer_win_index]))
    #     if filter_window//2 ==0:
    #         filter_window += 1
    #     #print('window size',filter_window)
    #     smoothed = savgol_filter(smoothed, window_length= filter_window, polyorder=3)

    #     filer_win_index += 1


    # print(noise_level,residual_noise,filer_win_index)




    # for i in range(3):
    #     filter_window_1 = max( int(len(Alg_y)/40),2) #in case data length is short
    #     if filter_window_1//2 ==0:
    #         filter_window_1 += 1
    #     Alg_y = savgol_filter(Alg_y,  filter_window_1, 2)

    #     filter_window_1 = max( int(len(Alg_y)/20),2) #in case data length is short
    #     if filter_window_1//2 ==0:
    #         filter_window_1 += 1
    #     Alg_y = savgol_filter(Alg_y,  filter_window_1, 2)

    #     filter_window_1 = max( int(len(Alg_y)/10),2) #in case data length is short
    #     if filter_window_1//2 ==0:
    #         filter_window_1 += 1
    #     Alg_y = savgol_filter(Alg_y,  filter_window_1, 2)

    # y_smooth_4 = savgol_filter(y_smooth_3,  filter_window_4, 2)


    # res_thre = 0.5
    # filer_win_factor = [15,10,3 ]
    # # filer_win_index = -1
    # filter_window_1 = int(len(Alg_y)/50)
    # if filter_window_1//2 ==0:
    #     filter_window_1 += 1
    # #print('window size',filter_window)
    # smoothed_1 = savgol_filter(Alg_y, window_length=filter_window_1 , polyorder=3)
    # residual_noise_1 = np.std(Alg_y-smoothed_1)

    # filter_window_2 = int(len(Alg_y)/25)
    # if filter_window_2//2 ==0:
    #     filter_window_2 += 1
    # #print('window size',filter_window)
    # smoothed_2 = savgol_filter(smoothed_1, window_length=filter_window_2 , polyorder=3)
    # residual_noise_2 = np.std(smoothed_2-smoothed_1)
    # # smoothed = smoothed_2
    # if (residual_noise_1-residual_noise_2)/residual_noise_1 < res_thre: 
    #     for filer_win_index in range(len(filer_win_factor) ):  # 遍历奇数窗口长度
             
    #         filter_window = int(len(Alg_y)/int(filer_win_factor[filer_win_index]))
    #         if filter_window//2 ==0:
    #             filter_window += 1
    #         #print('window size',filter_window)
    #         smoothed_3= savgol_filter(smoothed_2, window_length= filter_window, polyorder=3)
    #         residual_noise_3 = np.std(smoothed_3 - smoothed_2)
    #         if (residual_noise_2-residual_noise_3)/residual_noise_2 < res_thre:
    #             break
    #     smoothed = smoothed_3
    #     print(residual_noise_1,residual_noise_2,residual_noise_3)
    #     # print(residual_noise,filer_win_index)
    # else:
    #     smoothed = smoothed_2
    #     print(residual_noise_1,residual_noise_2)
    thre_region_len = Alg_Thre_Factor * len(Alg_X)
    dy_smooth = derivative(Alg_X, smoothed)  


    # #print(len(Alg_X),len(y_smooth_1),len(y_smooth_2))
    # degree = 3
    # # base_time = Sorted_Time[5:].min() 
    # coefficients = np.polyfit(Alg_X,y_smooth_2, degree)

    # poly_eq = np.poly1d(coefficients)
    # # x_fit = np.linspace(min(time_in_hours), max(time_in_hours), 100)
    # y_smooth_2 = poly_eq(Alg_X)



    # Ruptures for smoothed data
    len_baseline_fitting_smooth = len(Alg_X)
    num_bkps_smooth = min(4,int(len(Alg_y)/10))
    result_smooth = []
    while len_baseline_fitting_smooth > thre_region_len  and num_bkps_smooth > 1:
        num_bkps_smooth -= 1
        # algo_smooth = rpt.Dynp(model=model_cost).fit(dy_smooth)

        algo_smooth = get_algo_instance(SM, CF, dy_smooth)

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


