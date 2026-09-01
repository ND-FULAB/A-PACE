import csv
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
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
from natsort import natsorted
import queue
import matplotlib.animation as animation

LOGGER = logging.getLogger(__name__)


@dataclass
class RealTimeState:
    """Mutable history belonging to one real-time analysis task."""

    records: list[tuple[object, object, object, object]] = field(default_factory=list)
    cp_history: list[list[float]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _records_by_file: dict[str, list[tuple[object, object, object, object]]] = field(
        default_factory=dict, repr=False
    )
    _cp_by_file: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def add_records(self, times, peaks, peak_min, peak_max):
        with self.lock:
            self.records.extend(tuple(values) for values in zip(times, peaks, peak_min, peak_max))

    def add_cp_file(self, cp_values):
        mean = self._cp_mean(cp_values)
        if mean is None:
            return
        with self.lock:
            self.cp_history.append(mean)

    @staticmethod
    def _cp_mean(cp_values):
        valid = []
        for row in cp_values:
            try:
                pair = np.asarray(row, dtype=float).reshape(-1)[:2]
            except (TypeError, ValueError):
                continue
            if pair.size == 2 and np.all(np.isfinite(pair)):
                valid.append(pair)
        if not valid:
            return None
        return np.mean(np.asarray(valid), axis=0).tolist()

    def replace_file(self, source, times, peaks, peak_min, peak_max, cp_values):
        key = os.path.normcase(os.path.abspath(os.fspath(source)))
        records = [tuple(values) for values in zip(times, peaks, peak_min, peak_max)]
        cp_mean = self._cp_mean(cp_values)
        with self.lock:
            self._records_by_file[key] = records
            if cp_mean is None:
                self._cp_by_file.pop(key, None)
            else:
                self._cp_by_file[key] = cp_mean
            self._rebuild()

    def remove_file(self, source):
        key = os.path.normcase(os.path.abspath(os.fspath(source)))
        with self.lock:
            previous = (self._records_by_file.pop(key, None), self._cp_by_file.pop(key, None))
            self._rebuild()
            return previous

    def restore_file(self, source, previous):
        if previous == (None, None):
            return
        key = os.path.normcase(os.path.abspath(os.fspath(source)))
        records, cp_mean = previous
        with self.lock:
            if records is not None:
                self._records_by_file[key] = records
            if cp_mean is not None:
                self._cp_by_file[key] = cp_mean
            self._rebuild()

    def _rebuild(self):
        self.records = [
            record
            for file_records in self._records_by_file.values()
            for record in file_records
        ]
        self.cp_history = [
            self._cp_by_file[key]
            for key in self._records_by_file
            if key in self._cp_by_file
        ]

    def recent_cp_values(self, count):
        if count <= 0:
            return []
        with self.lock:
            return copy.deepcopy(self.cp_history[-count:])

    def snapshot_records(self):
        with self.lock:
            return list(self.records)

    def plot_payload(self):
        parsed = []
        for measured_at, peak, peak_min, peak_max in self.snapshot_records():
            try:
                parsed.append(
                    (parse_timestamp(measured_at), float(peak), float(peak_min), float(peak_max))
                )
            except (TypeError, ValueError):
                LOGGER.warning("Skipping real-time point with invalid values: %r", measured_at)
        parsed.sort(key=lambda item: item[0])
        if not parsed:
            return (), (), (), ()
        return tuple(zip(*parsed))

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
    if (
        Alg_raw.size == 0
        or Alg_raw.shape != Alg_baseline.shape
        or not np.all(np.isfinite(Alg_raw))
        or not np.all(np.isfinite(Alg_baseline))
    ):
        return False
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
    
    signal_range = np.ptp(Alg_raw)
    if sum_weights <= np.finfo(float).eps or signal_range <= np.finfo(float).eps:
        return False
    MWSE = MWSE / sum_weights / (signal_range**2)
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
    def __init__(
        self,
        Slip_Window,
        Alg_CPD_SM,
        Alg_CPD_CF,
        Alg_peak_region,
        Alg_noise_level,
        Fit_Alg,
        state=None,
    ):
        self.measurements = None
        self.curves = [[], []]
        self.times = []
        self.freq = []
        self.amp = []
        self.e_step = []
        self.channel = []
        self.len_curves = 0
        self.slip_window = max(1, int(Slip_Window))
        self.Alg_CPD_SM = Alg_CPD_SM
        self.Alg_CPD_CF = Alg_CPD_CF
        self.Alg_peak_region = Alg_peak_region
        self.Alg_noise_level = Alg_noise_level
        self.Fit_Alg = Fit_Alg
        self.state = state if state is not None else RealTimeState()
        self.last_raw_cp_values = []


    def read_pssession_file(self, filename):
        # Metadata belongs to one file. Reset it before every load so a missing
        # field in file B cannot reuse the value found in file A.
        self.measurements = None
        self.curves = [[], []]
        self.times = []
        self.freq = []
        self.amp = []
        self.e_step = []
        self.channel = []
        self.len_curves = 0

        self.measurements = list(
            pspyfiles.load_session_file(filename, load_peak_data=True, load_eis_fits=True)
        )
        self.curves[0] = [np.asarray(measurement.potential_arrays[0])[20:-20] for measurement in self.measurements]
        self.curves[1] = [np.asarray(measurement.current_arrays[0])[20:-20] for measurement in self.measurements]
        self.times = [getattr(measurement, "timestamp", None) for measurement in self.measurements]
        self.len_curves = len(self.measurements)

        try:
            with open(filename, encoding='utf-16le') as file:
                file_content = file.read()
        except (OSError, UnicodeError):
            LOGGER.exception("Could not read metadata from %s", filename)
            file_content = ""

        number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"

        def extract_float(key):
            return [float(value) for value in re.findall(re.escape(key) + number, file_content, re.IGNORECASE)]

        def align(values):
            # PalmSens files may repeat the method once outside measurements;
            # the last N entries correspond to the N measurement records.
            return (list(values)[-self.len_curves:] + [None] * self.len_curves)[:self.len_curves]

        self.e_step = align(extract_float('E_STEP='))
        self.freq = align(extract_float('FREQ='))
        self.amp = align(extract_float('E_AMP='))
        channels = [
            int(value)
            for value in re.findall(r'"channel"\s*:\s*(-?\d+)', file_content, re.IGNORECASE)
        ]
        self.channel = align(channels)
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
        Raw_CP_values = []
        for i in range( Num_Curves ):
            # print(Alg_File_Name,i,len(Alg_Data[0][i]),len(Alg_Data[1][i]))
            Curve_CP_index, Curve_CP_value,Alg_Current_CPD_smooth = Change_Point_Detection.CPD( Alg_Data[0][i],Alg_Data[1][i],self.Alg_CPD_SM,self.Alg_CPD_CF,self.Alg_peak_region,self.Alg_noise_level)

            # Curve_CP_index, Curve_CP_value = Change_Point_Detection.CPD( Alg_Data[0][i],Alg_Data[1][i],'Dynp','rank', 0.75)
            
            Raw_CP_values.append(Curve_CP_value)
            recent_cp = self.state.recent_cp_values(self.slip_window - 1)
            cp_window = recent_cp + [Curve_CP_value]
            potential = np.asarray(Alg_Data[0][i], dtype=float)
            if len(cp_window) >= self.slip_window:
                prev_CP_values = np.asarray(cp_window, dtype=float)
                avg_CP_value = np.mean(prev_CP_values, axis=0)  # 计算平均值
                closest_point_index_right = int(np.abs(potential - avg_CP_value[0]).argmin())
                closest_point_index_left = int(np.abs(potential - avg_CP_value[1]).argmin())
                Curve_CP_index_average = [closest_point_index_right, closest_point_index_left]
                Curve_CP_value_used = [
                    float(potential[closest_point_index_right]),
                    float(potential[closest_point_index_left]),
                ]
            else:
                Curve_CP_index_average = [int(Curve_CP_index[0]), int(Curve_CP_index[1])]
                Curve_CP_value_used = list(Curve_CP_value)

            Curve_CP_index_used = [max(Curve_CP_index_average), min(Curve_CP_index_average)]
            
            

            CP_index.append(Curve_CP_index_used)
            CP_value.append(Curve_CP_value_used)

            #dropout 
            if Curve_CP_index_used[0] == Curve_CP_index_used[1]:
                print(Alg_File_Name,i)
                print(Curve_CP_value_used, Curve_CP_index_used)
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
            mask[int(Curve_CP_index_used[1]):int(Curve_CP_index_used[0])] = 0
            weight = mask.astype(bool) 

            #Fit_Alg = ['imodpoly4', 'penalized_poly4', 'pspline_derpsalsa', 'pspline_iarpls', 'pspline_iasls', 'pspline_mpls', 'fabc']

            Alg_baselines = []
            Fit_Alg_using = copy.copy(self.Fit_Alg)  #this list is to store the using algs

            index_baseline_fitting_standard = [] #index for baseline not satisfiled baseline_fitting_standard()        
            for fitting_alg in self.Fit_Alg:
                (baseline, para), error = get_algo_instance(fitting_alg,Alg_Data[0][i],Alg_Current_CPD_smooth,Fit_Order,Num_Iter,weight)
                if error:
                    print(Alg_File_Name,self.Fit_Alg,error)

                elif baseline_fitting_standard(Curve_CP_index_used, Alg_Current_CPD_smooth, baseline):
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

                peak_left = int(Curve_CP_index_used[1])
                peak_right = int(Curve_CP_index_used[0])
                Curve_Peak_Mean,Curve_Peak_Location,Curve_Peak_Index = peak_info(
                    Alg_Data[0][i][peak_left:peak_right],
                    [a - b for a, b in zip(Alg_Current_CPD_smooth, Curve_Baseline_Mean)][peak_left:peak_right],
                )
                Peak_Mean.append(Curve_Peak_Mean)
                Peak_Location.append(Curve_Peak_Location)
                print(Alg_File_Name, Curve_Peak_Index, peak_left)
                Peak_Min.append(Curve_Peak_Mean - Curve_Baseline_CI[Curve_Peak_Index + peak_left])
                Peak_Max.append(Curve_Peak_Mean + Curve_Baseline_CI[Curve_Peak_Index + peak_left])


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
        self.last_raw_cp_values = Raw_CP_values
        return Alg_File_Name ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location,Alg_index 
      

def parse_timestamp(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if ts is None:
        raise ValueError("Missing measurement timestamp")
    ts = str(ts).strip()
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
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        pass
    # if nothing matched, raise or fallback
    raise ValueError(f"Unrecognized datetime format: {ts!r}")
    
def _is_pssession(path):
    return os.path.splitext(os.fspath(path))[1].lower() == ".pssession"


def _file_signature(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def wait_for_file_stable(path, timeout=5.0, interval=0.25, stable_observations=2):
    """Wait until both size and mtime are unchanged in consecutive samples."""
    deadline = time.monotonic() + timeout
    previous = None
    stable_count = 0
    while time.monotonic() < deadline:
        try:
            stat = os.stat(path)
            signature = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            signature = None
        if signature is not None and signature == previous:
            stable_count += 1
            if stable_count >= stable_observations:
                return True
        else:
            stable_count = 0
        previous = signature
        time.sleep(interval)
    return False


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _atomic_json_dump(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".apace-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(_json_compatible(data), output, indent=4, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class RealTimeAnalysis(FileSystemEventHandler):
    def __init__(self, plot_queue, processor, state=None, stability_checker=wait_for_file_stable):
        self.plot_queue = plot_queue
        self.processor = processor
        self.state = state if state is not None else processor.state
        self.stability_checker = stability_checker
        self.file_index = 0
        self._paths_lock = threading.Lock()
        self._pending_paths = set()
        self._processed_signatures = {}
        print('Start detecting')

    def on_created(self, event):
        self._handle_event(event.src_path, event.is_directory)

    def on_modified(self, event):
        self._handle_event(event.src_path, event.is_directory)

    def on_moved(self, event):
        self._handle_event(event.dest_path, event.is_directory)

    def _handle_event(self, path, is_directory=False):
        if is_directory or not _is_pssession(path):
            return
        normalized = os.path.normcase(os.path.abspath(path))
        signature = _file_signature(path)
        with self._paths_lock:
            if normalized in self._pending_paths:
                return
            if signature is not None and self._processed_signatures.get(normalized) == signature:
                return
            self._pending_paths.add(normalized)
        try:
            if not self.stability_checker(path):
                raise TimeoutError(f"File did not stabilize within five seconds: {path}")
            stable_signature = _file_signature(path)
            if stable_signature is None:
                raise FileNotFoundError(path)
            with self._paths_lock:
                if self._processed_signatures.get(normalized) == stable_signature:
                    return
            self.process_new_file(path)
        except Exception:
            # One corrupt or partially written file must not terminate watchdog's
            # dispatcher thread. A later modified event is allowed to retry it.
            LOGGER.exception("Failed to process real-time file %s", path)
        else:
            with self._paths_lock:
                # Record the exact version that was deemed stable before parsing.
                # If the instrument wrote again during parsing, the queued modified
                # event will carry a different signature and trigger replacement.
                self._processed_signatures[normalized] = stable_signature
        finally:
            with self._paths_lock:
                self._pending_paths.discard(normalized)

    def process_new_file(self, file_path):
        print(f"Processing file: {file_path}")
        previous = self.state.remove_file(file_path)
        try:
            curves, times, len_curves, freq, amp, e_step, channel = self.processor.read_pssession_file(file_path)
            results = self.processor.process_file(file_path, curves, len_curves, self.file_index)

            data_save = {}
            for curve_index in range(len_curves):
                data_save['Curve No. '+str(curve_index+1)] = {
                    'Date and time measurement': times[curve_index],
                    'Channel ': channel[curve_index],
                    'Frequence ': freq[curve_index],
                    'Amplitude ': amp[curve_index],
                    'E Step ': e_step[curve_index],
                    'Raw Poetntial ': curves[0][curve_index],
                    'Raw Current': curves[1][curve_index],
                    'Change Point Indexes ': results[1][curve_index],
                    'Change Point Values ': results[2][curve_index],
                    'Baseline Mean ': results[3][curve_index],
                    '99\\% Confidence Interval of Baseline: ': results[4][curve_index],
                    'Peak Value ': results[5][curve_index],
                    '99\\% Confidence Interval of Peak Value': [
                        results[7][curve_index], results[6][curve_index]
                    ],
                    'Peak Location: ': results[8][curve_index],
                }

            directory, filename = os.path.split(os.path.normpath(file_path))
            path2save = os.path.join(directory, 'APACE_result')
            result_path = os.path.join(path2save, filename + '_result.json')
            _atomic_json_dump(result_path, data_save)
        except Exception:
            self.state.restore_file(file_path, previous)
            raise

        self.file_index += 1
        self.state.replace_file(
            file_path,
            times,
            results[5],
            results[7],
            results[6],
            self.processor.last_raw_cp_values,
        )
        payload = self.state.plot_payload()
        if payload[0]:
            self.plot_queue.put(payload)

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
        self._save_timer = None
        self._periodic_state = None
        self._periodic_plot_path = None
        self._periodic_data_path = None

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
            self.ax.margins(x=0, y=0.05)
        return self.line,

    def save_plot(self, filename):
        self.fig.savefig(filename)
        print(f"Figure saved as {filename}")

    def save_data(self, filename, data):
        absolute = os.path.abspath(filename)
        directory = os.path.dirname(absolute)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".apace-", suffix=".csv", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["Time", "Peak Mean", "Peak Min", "Peak Max"])
                writer.writerows(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, absolute)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        print(f"Data saved as {filename}")

    def start_periodic_save(self, state, plot_path, data_path, interval_ms=20_000):
        self._periodic_state = state
        self._periodic_plot_path = plot_path
        self._periodic_data_path = data_path
        self._save_timer = self.fig.canvas.new_timer(interval=interval_ms)
        self._save_timer.add_callback(self._periodic_save)
        self._save_timer.start()

    def _periodic_save(self):
        try:
            self.save_plot(self._periodic_plot_path)
            self.save_data(self._periodic_data_path, self._periodic_state.snapshot_records())
        except Exception:
            LOGGER.exception("Could not save periodic real-time output")

    def stop_periodic_save(self):
        if self._save_timer is not None:
            self._save_timer.stop()
            self._save_timer = None

def run_real_time_analysis(path,slip_window,CPD_SM,CPD_CF,Alg_peak,Alg_noise_level, Alg_fitting_set ):
    if not os.path.isdir(path):
        raise ValueError(f"Real-time folder does not exist: {path}")

    state = RealTimeState()
    plot_queue = queue.Queue()
    processor = DataProcess(
        slip_window,
        CPD_SM,
        CPD_CF,
        Alg_peak,
        Alg_noise_level,
        Alg_fitting_set,
        state=state,
    )
    event_handler = RealTimeAnalysis(plot_queue, processor, state=state)

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    plotter = None
    try:
        plotter = RealTimePlotter(plot_queue)
        plotter.start_periodic_save(state, "output_plot.png", "output_data_temp.csv")
        # Matplotlib owns the GUI event loop. Its timer performs periodic saves;
        # closing the window returns from show instead of entering an infinite loop.
        plt.show(block=True)
    finally:
        print("Stopping observer...")
        observer.stop()
        observer.join()
        if plotter is not None:
            plotter.stop_periodic_save()
            try:
                plotter.save_plot("output_plot.png")
                plotter.save_data("output_data.csv", state.snapshot_records())
            except Exception:
                LOGGER.exception("Could not save final real-time output")
            plt.close(plotter.fig)
        print("Observer stopped. Exiting program.")



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



