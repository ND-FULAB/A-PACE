import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import Change_Point_Detection
import pspython.pspyfiles as pspyfiles
from multiprocessing import Pool,cpu_count,freeze_support
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
# import matplotlib
# matplotlib.use('Agg')
import matplotlib
matplotlib.use('TkAgg')  # 使用支持交互式绘图的后端
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.pyplot as plt
import ast
import numpy as np
from datetime import datetime
from Algs import get_algo_instance
import re
import matplotlib.dates as mdates
import json
import os
import copy
import math
import time
from natsort import natsorted
import queue
import matplotlib.animation as animation

all_times  = []
all_peaks = []
all_peak_min = []  
all_peak_max = []
#this array is for all calculated cp values from Change_Point_Detection.CPD(), while peak info is based on averaged cp vaules
all_cp_values = [[],[]] 

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def baseline_fitting_standard(  Alg_Boundarys, Alg_Raw_Current,Alg_Baseline_Current ):
    

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



def extreme_baseline_detection(  Alg_Baselines):

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
    ci =  [(a - b)/2 for a, b in zip(ci_upper.tolist(), ci_lower.tolist())]

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
    ci =  [(a - b)/2 for a, b in zip(ci_upper.tolist(), ci_lower.tolist())]  
    return median_values.tolist(), ci



def peak_info(  Alg_Potential, Alg_Current ):
    # peak_smooth = savgol_filter(Alg_Current, 5, 2)
    peak_index = np.argmax( Alg_Current) 
    peak_location = Alg_Potential[   peak_index ]

    return   np.max( Alg_Current), peak_location, peak_index




class DataProcess:
    def __init__(self,Slip_Window,Alg_CPD_SM,Alg_CPD_CF,Alg_peak_region,Alg_noise_level,Fit_Alg):
        self.measurements = None
        self.curves = [[], []]
        self.times = []
        self.freq = []
        self.amp = []
        self.e_step = []
        self.channel = []
        self.len_curves = 0
        self.slip_window = Slip_Window 
        self.Alg_CPD_SM = Alg_CPD_SM
        self.Alg_CPD_CF = Alg_CPD_CF
        self.Alg_peak_region = Alg_peak_region
        self.Alg_noise_level = Alg_noise_level
        self.Fit_Alg = Fit_Alg


    def read_pssession_file(self, filename):
        self.measurements = pspyfiles.load_session_file(filename, load_peak_data=True, load_eis_fits=True)
        self.curves[0] = [measurement.potential_arrays[0][20:-20] for measurement in self.measurements]
        self.curves[1] = [measurement.current_arrays[0][20:-20] for measurement in self.measurements]
        self.times = [measurement.timestamp for measurement in self.measurements]
        self.len_curves = len(self.measurements)
        
        keyword = 'E_STEP='
        counter = copy.copy(self.len_curves)
        with open(filename, encoding='utf-16le') as file:
            file_content = file.readlines()
        file_content = file_content[0]
        for i in range(len(file_content) - 7):
            if file_content[i:i + 7] == keyword:
                if counter == self.len_curves:
                    counter -= 1
                else:
                    self.e_step.append(float(file_content[i + 7:i + 17]))
                    for j in range(i + 200, i + 2000):
                        if file_content[j:j + 5] == 'FREQ=':
                            self.freq.append(float(file_content[j + 5:j + 15]))
                            self.amp.append(float(file_content[j + 25:j + 35]))

        key_forsearch = '"channel'
        matches = list(re.finditer(re.escape(key_forsearch), file_content, re.IGNORECASE))
        count = len(matches)
        interval = int(count / self.len_curves)
        positions = [match.start() for match in matches]
        start = 1 if interval > 1 else 0
        for i in range(start, count, interval):
            for j in range(positions[i], positions[i] + 100):
                if file_content[j] == ',':
                    substring = file_content[positions[i]:j]
                    num_instring = re.findall(r'\d+', substring)
                    self.channel.append(int(num_instring[0]))
                    break
        return self.curves, self.times, self.len_curves, self.freq, self.amp, self.e_step, self.channel



    def process_file(self,Alg_File_Name, Alg_Data, Num_Curves,Alg_index):

        # folder_path = './Fig_Saved'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path,exist_ok=True)

        CP_index = []
        CP_value = [] 
        Baseline_Mean = [] 
        Baseline_CI = []
        Peak_Mean = []
        Peak_Max = []
        Peak_Min = []
        Peak_Location = []
        for i in range( Num_Curves ):
            # print(Alg_File_Name,i,len(Alg_Data[0][i]),len(Alg_Data[1][i]))
            Curve_CP_index, Curve_CP_value,Alg_Current_CPD_smooth = Change_Point_Detection.CPD( Alg_Data[0][i],Alg_Data[1][i],self.Alg_CPD_SM,self.Alg_CPD_CF,self.Alg_peak_region,self.Alg_noise_level)

            # Curve_CP_index, Curve_CP_value = Change_Point_Detection.CPD( Alg_Data[0][i],Alg_Data[1][i],'Dynp','rank', 0.75)
            
            if Alg_index >= self.slip_window:
                prev_CP_values = all_cp_values[-self.slip_window+1:]
                prev_CP_values.append( Curve_CP_value )  
                prev_CP_values = np.array(prev_CP_values)
                avg_CP_value = np.mean(prev_CP_values, axis=0)  # 计算平均值
                closest_point_index_right = (np.abs(Alg_Data[0][i] - avg_CP_value[0])).argmin()  
                closest_point_index_left = (np.abs(Alg_Data[0][i] - avg_CP_value[1])).argmin() 
                Curve_CP_index_average = [closest_point_index_right, closest_point_index_left]
            else:
                Curve_CP_index_average = Curve_CP_index
            
            

            CP_index.append(Curve_CP_index)
            CP_value.append(Curve_CP_value)

            #dropout 
            if Curve_CP_index[0] == Curve_CP_index[1]:
                print(Alg_File_Name,i)
                print( Curve_CP_value, Curve_CP_index )
                Baseline_Mean.append([])
                Baseline_CI.append([])
                Peak_Mean.append(0) 
                Peak_Max.append(0) 
                Peak_Min.append(0) 
                Peak_Location.append(0)


                continue

            Fit_Order = 3
            Num_Iter = 9999
            mask = np.ones(shape = len(Alg_Data[0][i]))
            mask[ int(Curve_CP_index[1]): int(Curve_CP_index[0])] = 0#be consist with boundary calculation 
            weight = mask.astype(bool) 

            #Fit_Alg = ['imodpoly4', 'penalized_poly4', 'pspline_derpsalsa', 'pspline_iarpls', 'pspline_iasls', 'pspline_mpls', 'fabc']

            Alg_baselines = []
            Fit_Alg_using = copy.copy(self.Fit_Alg)  #this list is to store the using algs

            index_baseline_fitting_standard = [] #index for baseline not satisfiled baseline_fitting_standard()        
            for fitting_alg in self.Fit_Alg:
                (baseline, para), error = get_algo_instance(fitting_alg,Alg_Data[0][i],Alg_Current_CPD_smooth,Fit_Order,Num_Iter,weight)
                if error:
                    print(Alg_File_Name,self.Fit_Alg,error)

                elif baseline_fitting_standard(  Curve_CP_index, Alg_Current_CPD_smooth,baseline  ):
                    # print(fitting_alg,'works')
                    Alg_baselines.append(baseline)
                    #plt.plot(Alg_Data[0][i],baseline, label=fitting_alg)
                else:
                    Fit_Alg_using.remove( fitting_alg  )
                    #plt.plot(Alg_Data[0][i],baseline, '--', label=fitting_alg)
                    
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
                # plt.savefig('Fig_Saved/'+ Alg_File_Name + '_'+str(i) + 'alg.png')
                # plt.close('all')
                continue

            if len(Alg_baselines) <5:  #no extreme value detection
                Curve_Baseline_Mean, Curve_Baseline_CI =get_CI(Alg_baselines)
                Baseline_Mean.append(Curve_Baseline_Mean)
                Baseline_CI.append( Curve_Baseline_CI )
            else:
                Curve_Baseline_Mean, Curve_Baseline_CI,Curve_Outlier_Info =  extreme_baseline_detection(Alg_baselines)
                Baseline_Mean.append(Curve_Baseline_Mean)
                Baseline_CI.append( Curve_Baseline_CI )

            if Curve_Baseline_Mean:

                Curve_Peak_Mean,Curve_Peak_Location,Curve_Peak_Index = peak_info( Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a - b for a, b in zip(Alg_Current_CPD_smooth, Curve_Baseline_Mean)][ int(Curve_CP_index[1]):int(Curve_CP_index[0])]  )
                Peak_Mean.append(Curve_Peak_Mean)
                Peak_Location.append(Curve_Peak_Location)
                print(Alg_File_Name,Curve_Peak_Index , int(Curve_CP_index[1]))
                Peak_Min.append(Curve_Peak_Mean - Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ] )
                Peak_Max.append(Curve_Peak_Mean +  Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ])


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
                # plt.savefig('Fig_Saved/'+ Alg_File_Name + '_'+str(i) + 'alg.png')
                # plt.close('all')


        return Alg_File_Name ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location,Alg_index 
      

from datetime import datetime
def parse_timestamp(ts: str) -> datetime:
    for fmt in (
        "%Y-%m-%d %H:%M:%S",      # 2025-05-09 14:01:44
        "%Y/%m/%d %H:%M:%S",      # 2025/05/09 14:01:44
        "%Y-%m-%d %I:%M:%S %p",   # 2025-05-09 02:01:44 PM
        "%Y/%m/%d %I:%M:%S %p",   # 2025/05/09 02:01:44 PM
        "%m/%d/%Y %I:%M:%S %p",   # 4/4/2025  3:37:39 PM
        "%m/%d/%Y %H:%M:%S",      # 4/4/2025 15:37:39
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    # if nothing matched, raise or fallback
    raise ValueError(f"Unrecognized datetime format: {ts!r}")
    
class RealTimeAnalysis(FileSystemEventHandler):
    def __init__(self, plot_queue, processor):
        self.plot_queue = plot_queue
        self.processor = processor
        self.file_index = 0 
        print('Start detecting')
        


    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New file detected: {event.src_path}")
        time.sleep(0.2)#make sure the file is complete
        self.process_new_file(event.src_path)

    def process_new_file(self, file_path):
        print(f"Processing file: {file_path}")
        curves, times, len_curves, freq, amp, e_step, channel = self.processor.read_pssession_file(file_path)
        results = self.processor.process_file(file_path, curves, len_curves,self.file_index)

        data_save = {}
        for curve_index in range(len_curves):
            data_save['Curve No. '+str(curve_index+1)] = {
                            'Date and time measurement': times[curve_index],
                            'Channel ': channel[curve_index],
                            'Frequence ': freq[curve_index],
                            'Amplitude ': amp[curve_index],
                            'E Step ': e_step[curve_index],
                            'Raw Poetntial ': curves[0][curve_index],
                            'Raw Current':curves[1][curve_index],
                            'Change Point Values ' : results[1][curve_index],

                'Baseline Mean ': results[3][curve_index],                
                '95\\% Confidence Interval of Baseline: ' : results[4][curve_index],  
                 'Peak Value ' : results[5][curve_index], 
                 '95\\% Confidence Interval of Peak Value' : [results[7][curve_index], results[6][curve_index] ],
                 'Peak Location: ' :results[8][curve_index] 


            }
        full_path = os.path.normpath(file_path)

        # Split into directory and filename
        directory, filename = os.path.split(full_path)

        folder2save = 'APACE_result'
        path2save = os.path.join(directory, folder2save)
        if not os.path.exists(path2save):

            os.makedirs(path2save,exist_ok=True)


        with open(path2save+'/' + filename +'_result.json', 'w') as json_file:
            json.dump(data_save, json_file,indent=4,  ensure_ascii=False)


        self.file_index += 1
        all_times.append(times)
        all_peaks.append(results[5])
        #print(times,results[5])
        all_peak_min.append(results[7]) 
        all_peak_max.append(results[6]) 
        CP_file = np.array( results[2] )
        cp_average = np.mean(CP_file,axis=0)
        all_cp_values.append( list(cp_average) )
        # cp_right = [ row[0] for row in results[2]   ]
        # cp_left = [ row[1] for row in results[2]   ]
        # all_cp_values[0].append( sum(cp_right)/len(cp_right)  )
        # all_cp_values[1].append( sum(cp_left)/len(cp_left)  )

        time2plot = copy.copy(all_times)
        peak2plot = copy.copy(all_peaks)
        peak_min2plot = copy.copy(all_peak_min)
        peak_max2plot = copy.copy(all_peak_max)

        time2plot = flatten_list(time2plot)
        peak2plot = flatten_list(peak2plot)
        peak_min2plot = flatten_list(peak_min2plot)
        peak_max2plot = flatten_list(peak_max2plot)

        # time2plot = [datetime.strptime(t, '%Y/%m/%d %H:%M:%S') for t in time2plot]
        time2plot = [parse_timestamp(t) for t in time2plot]

        combined_list = list(zip(time2plot, peak2plot, peak_min2plot, peak_max2plot))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        Sorted_Time, Sorted_Peak, Sorted_Peak_Min, Sorted_Peak_Max = zip(*sorted_combined_list)

        self.plot_queue.put((Sorted_Time, Sorted_Peak, Sorted_Peak_Min, Sorted_Peak_Max))

class RealTimePlotter:
    def __init__(self, plot_queue):
        self.plot_queue = plot_queue
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Real-Time Data Plot")
        self.ax.set_xlabel("Time (H:M:S)")
        self.ax.set_ylabel("Peak Mean")
        self.line, = self.ax.plot([], [], 'b-', label='Peak Mean')
        self.scatter = self.ax.scatter([], [], c='red')  
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        self.fill_between = None

    def update_plot(self, frame):
        while not self.plot_queue.empty():
            
            Plotter_Time, Plotter_Peak, Plotter_Peak_Min, Plotter_Peak_Max = self.plot_queue.get()
            #print('Time4plot : ',Plotter_Time )
            times_array = np.array(Plotter_Time)
            peak_means_array = np.array(Plotter_Peak)
            peak_min_array = np.array(Plotter_Peak_Min)
            peak_max_array = np.array(Plotter_Peak_Max)

            self.line.set_xdata(times_array)
            self.line.set_ydata(peak_means_array)
            self.scatter.set_offsets(np.c_[times_array, peak_means_array])  

            if self.fill_between is not None:
                self.fill_between.remove()

            self.fill_between = self.ax.fill_between(
                times_array,
                peak_min_array,
                peak_max_array,
                color='blue', alpha=0.2, label='Peak Range'
            )

            self.ax.relim()#recalculate the axis range
            self.ax.autoscale_view()
            # self.ax.margins(max(peak_max_array)) 
            self.ax.margins(x=0, y=max(peak_max_array))
        return self.line,

    def save_plot(self, filename):
        self.fig.savefig(filename)
        print(f"Figure saved as {filename}")

    def save_data(self, filename, data):
        np.savetxt(filename, data, delimiter=",", fmt="%s", header="Time,Peak Mean,Peak Min,Peak Max")
        print(f"Data saved as {filename}")

def run_real_time_analysis(path,slip_window,CPD_SM,CPD_CF,Alg_peak,Alg_noise_level, Alg_fitting_set ):
    plot_queue = queue.Queue()
    processor = DataProcess(slip_window,CPD_SM,CPD_CF,Alg_peak,Alg_noise_level, Alg_fitting_set)
    event_handler = RealTimeAnalysis(plot_queue, processor)

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    plotter = RealTimePlotter(plot_queue)
    try:
        plt.show(block=True)  # 阻塞窗口，确保图像显示
        last_save_time = time.time()
        save_interval = 20  # 每隔60秒保存一次
        while True:
            time.sleep(.001)
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                print("Saving plot and data...")
                plotter.save_plot("output_plot.png")  # 定时保存图像
                # Plotter_Time, Plotter_Peak, Plotter_Peak_Min, Plotter_Peak_Max = plot_queue.get()
                combined_data = np.column_stack((all_times,all_peaks,all_peak_min ,all_peak_max ))
                plotter.save_data("output_data_temp.csv", combined_data)  
                last_save_time = current_time

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, saving plot and data...")
        plotter.save_plot("output_plot.png")
        plt.close()

        combined_data = np.column_stack((all_times,all_peaks,all_peak_min ,all_peak_max ))
        plotter.save_data("output_data.csv", combined_data) 
    finally:
        print("Stopping observer...")
        observer.stop() 
        observer.join()  
        print("Observer stopped. Exiting program.")
    # print()



if __name__ == "__main__":
    import sys, ast

    # We expect exactly 7 args, in this order:
    #   1. folder path
    #   2. slip window    (int)
    #   3. CPD search model (str)
    #   4. CPD cost function(str)
    #   5. peak ratio / threshold (float)
    #   6. noise level    (int)
    #   7. fitting alg list (Python literal list)
    if len(sys.argv) != 8:
        print("Usage: real_time_analysis.py <folder> <slip> <CPD_SM> <CPD_CF> "
              "<peak_ratio> <noise_level> <fitting_list>")
        sys.exit(1)

    folder       = sys.argv[1]
    slip_window  = int(sys.argv[2])
    CPD_SM       = sys.argv[3]
    CPD_CF       = sys.argv[4]
    peak_ratio   = float(sys.argv[5])
    noise_level  = int(sys.argv[6])
    fitlist      = ast.literal_eval(sys.argv[7])

    # —— Debug prints —— 
    print("=== real_time_analysis.py arguments ===")
    print(f"  folder      = {folder!r}")
    print(f"  slip_window = {slip_window!r}")
    print(f"  CPD_SM      = {CPD_SM!r}")
    print(f"  CPD_CF      = {CPD_CF!r}")
    print(f"  peak_ratio  = {peak_ratio!r}")
    print(f"  noise_level = {noise_level!r}")
    print(f"  fitlist     = {fitlist!r}")
    print("=======================================")

    # Now call your function with exactly those seven arguments:
    run_real_time_analysis(
        folder,
        slip_window,
        CPD_SM,
        CPD_CF,
        peak_ratio,
        noise_level,
        fitlist
    )

#     SR_weight = 1.0

#     with open('Algorithm Setting.json', 'r', encoding='utf-8') as file:
#         data_algs = json.load(file)
#     name = str(int(SR_weight/0.01))


#     # names = list(data_algs.keys())
#     # for alg_index in names:
#     #     if data_algs[alg_index]["Success Weight"] == round(SR_weight,2):
#     #         break
#     noise_level = 3
#     path = "C:\\Users\\jyk98\\Desktop\\Basepeak\\Basepeak-New\\real_time\\path"  # 替换为你要监控的文件夹路径
#     run_real_time_analysis(path,5,
#                   data_algs[name][ "CPD Search Model"],
#                   data_algs[name][ "CPD Cost Function"],
#                   float(data_algs[name][ "Ratio For Peak"]),
#                   noise_level,
#                   list(ast.literal_eval(data_algs[name][ "Baseline Fitting Algorithms"]) )
                           
                           
#                            )



