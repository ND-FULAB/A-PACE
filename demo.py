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
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import ast
import codecs
from storage import RESULTS_PATH, write_json


CSV_HEADER_MARKER = "Date and time measurement:"
CSV_ENCODING_SAMPLE_SIZE = 64 * 1024


def _file_decodes_as(filename, encoding):
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    try:
        with open(filename, "rb") as file:
            while chunk := file.read(CSV_ENCODING_SAMPLE_SIZE):
                decoder.decode(chunk, final=False)
        decoder.decode(b"", final=True)
    except UnicodeError:
        return False
    return True


def detect_csv_encoding(filename):
    """Detect encodings used by APACE/PalmSens CSV exports."""

    with open(filename, "rb") as file:
        sample = file.read(CSV_ENCODING_SAMPLE_SIZE)

    if not sample:
        raise ValueError(f"CSV file is empty: {filename}")

    # Check UTF-32 before UTF-16 because the little-endian BOMs share a prefix.
    bom_encodings = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )
    for bom, encoding in bom_encodings:
        if sample.startswith(bom):
            if not _file_decodes_as(filename, encoding):
                raise UnicodeError(
                    f"CSV BOM indicates {encoding}, but the file contains "
                    "invalid data for that encoding."
                )
            return encoding

    # UTF-16 without a BOM can also look like valid UTF-8 because its NUL bytes
    # are legal ASCII controls. The APACE header disambiguates its byte order.
    for encoding in ("utf-16-le", "utf-16-be"):
        marker = CSV_HEADER_MARKER.encode(encoding)
        marker_offset = sample.find(marker)
        if (
            marker_offset >= 0
            and marker_offset % 2 == 0
            and _file_decodes_as(filename, encoding)
        ):
            return encoding

    # ASCII-only Windows-1252 is valid UTF-8 as well, so prefer UTF-8 whenever
    # the complete file decodes strictly. Otherwise try the Windows fallback.
    for encoding in ("utf-8", "cp1252"):
        if _file_decodes_as(filename, encoding):
            return encoding

    raise UnicodeError(
        f"Could not determine the encoding of {filename}. Supported encodings "
        "are UTF-8, UTF-16, BOM-marked UTF-32, and Windows-1252."
    )


def read_csv_file(filename):
    print(filename)
    encoding = detect_csv_encoding(filename)
    with open(filename, encoding=encoding) as file:
        data = file.readlines()
    if len(data) < 6:
        raise ValueError("CSV file does not contain the expected APACE header.")
    time_info = data[4]
    pattern = r"Date and time measurement:,\s*([\d/: -]+)"
    raw_dates = re.findall(pattern, time_info)
    if not raw_dates:
        raise ValueError("CSV file does not contain a measurement date.")

# 解析每个找到的日期时间字符串，确保解析顺序与出现顺序一致
    parsed_dates = [parser.parse(date) for date in raw_dates]
    len_curves = len(parsed_dates)
    df = pd.read_csv(filename, encoding=encoding, skiprows=6, header=None)
    required_columns = len_curves * 2
    if df.shape[1] < required_columns:
        raise ValueError(
            f"CSV file has {df.shape[1]} data columns; expected at least "
            f"{required_columns} for {len_curves} curve(s)."
        )
    #print(df)
    result = df.values.tolist()
    if len(result) <= 41:
        raise ValueError("CSV file must contain more than 41 data rows.")
    #print(result)
    curves = [[[] for _ in range(len_curves)],[[] for _ in range(len_curves)]]

    curve_data = result[20:-21]
    for i in range(len_curves):
        for row in curve_data:
            curves[0][i].append(float(row[i * 2]))
            curves[1][i].append(float(row[1 + i * 2]) )


    return(curves,parsed_dates, len_curves)

# SDK only

def read_pssession_file( filename):
    try:
        Measurements = pspyfiles.load_session_file(
            filename, load_peak_data=True, load_eis_fits=True
        )
        len_curves = len(Measurements)
    except Exception as error:
        raise ValueError(f"Could not read pssession file {filename}: {error}") from error
    if len_curves == 0:
        raise ValueError(f"Pssession file contains no measurements: {filename}")

    curves = [[],[]]
    # print(len_curves)

    freq =np.zeros(len_curves)
    amp = np.zeros(len_curves)
    e_step = np.zeros(len_curves)
    channel = []

    curves[0] =  [measurement.potential_arrays[0][20:-20]for measurement in Measurements]
    curves[1] =  [measurement.current_arrays[0][20:-20]for measurement in Measurements]
    times=  [measurement.timestamp for measurement in Measurements]

    keyword = 'E_STEP='
    # keyword = keyword.encode('utf-16le')

    counter = -1
    with open(filename, encoding='utf-16le') as file:
        file_content = file.readlines()
    file_content= file_content[0]
    for i in range(  len(file_content)-7   )  :
        if file_content[i:i+7] == keyword :
            if counter == len_curves:
                break
            if counter != -1:
                # print(i,'location')
                # print(len(file_content))
                e_step[counter] = float(file_content[i+7:i+17])

                for j in range( i+200,i+2000 ):
                    if file_content[j:j+5] == 'FREQ=':
                        freq[counter] = float(file_content[j+5:j+15])
                        amp[counter] =  float(file_content[j+25:j+35])
            counter += 1
    key_forsearch = '"channel'
    # pattern = rf'“[^”]*{ key_forsearch}[^”]*$'
    matches = list(re.finditer(re.escape(key_forsearch), file_content, re.IGNORECASE))
    count = len(matches)

    interval = int(count/len_curves)
    if interval == 0:
        for i in range(len_curves):
            channel.append(0)
    else:
        positions = [match.start() for match in matches]
        if interval >1 : #when tittle include "channel, tittle appears first in positions list
            start = 1
        else:
            start = 0
        for i in range( start,count,interval ):
            for j in range( positions[i],positions[i]+100 ):
                if file_content[j] == ',':
                    substring = file_content[positions[i]:  j]

                    num_instring = re.findall(r'\d+', substring)
                    channel.append( int( num_instring[0] ) )
                    break
    # print(channel)
    return (curves,times,len_curves,freq.tolist(),amp.tolist(),e_step.tolist(), channel )


def baseline_fitting_standard(  Alg_Boundarys, Alg_Raw_Current,Alg_Baseline_Current ):
    Alg_Raw_Current = np.asarray(Alg_Raw_Current, dtype=float)
    Alg_Baseline_Current = np.asarray(Alg_Baseline_Current, dtype=float)
    if (
        Alg_Raw_Current.ndim != 1
        or Alg_Baseline_Current.ndim != 1
        or len(Alg_Raw_Current) != len(Alg_Baseline_Current)
        or len(Alg_Raw_Current) == 0
        or not np.all(np.isfinite(Alg_Raw_Current))
        or not np.all(np.isfinite(Alg_Baseline_Current))
    ):
        return False
    try:
        upper, lower = (int(Alg_Boundarys[0]), int(Alg_Boundarys[1]))
    except (TypeError, ValueError, IndexError):
        return False
    if not 0 <= lower < upper < len(Alg_Raw_Current):
        return False
    Alg_Boundarys = (upper, lower)
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

    area_tolerance = (
        np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(Alg_Raw_Current))))
        * (Alg_Boundarys[0] - Alg_Boundarys[1])
    )
    if Alg_area_linear > area_tolerance:
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
    if len(Alg_raw) == 0:
        return False
    signal_range = np.ptp(Alg_raw)
    if not np.isfinite(signal_range) or signal_range == 0:
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
    if sum_weights == 0:
        return False

    MWSE =  0

    for point_index in range( len(Alg_raw)  ):
        MWSE += weights[point_index] * Square_Error[point_index]

    MWSE = MWSE / sum_weights / (signal_range ** 2)
    if MWSE > 0.1:
        print('MWSE: ', MWSE)
    #print(pearson_r,p_value)
    return MWSE < 0.1


def extreme_baseline_detection(  Alg_Baselines):
    Alg_Baselines = np.asarray(Alg_Baselines, dtype=float)
    if (
        Alg_Baselines.ndim != 2
        or Alg_Baselines.shape[0] == 0
        or Alg_Baselines.shape[1] == 0
        or not np.all(np.isfinite(Alg_Baselines))
    ):
        raise ValueError("baselines must be a non-empty finite 2D array")
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
    if len(non_outliers) == 0:
        non_outliers = Alg_Baselines
        non_outliers_mask = np.ones(len(Alg_Baselines), dtype=bool)

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
    Alg_Baselines = np.asarray(Alg_Baselines, dtype=float)
    if (
        Alg_Baselines.ndim != 2
        or Alg_Baselines.shape[0] == 0
        or Alg_Baselines.shape[1] == 0
        or not np.all(np.isfinite(Alg_Baselines))
    ):
        raise ValueError("baselines must be a non-empty finite 2D array")
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


PEAK_WIDTH_KEY = "Peak Width at Half Maximum"


def _half_height_crossing(potential, current, peak_index, half_height, direction):
    """Return the nearest interpolated half-height crossing in ``direction``."""
    if direction < 0:
        segment_indexes = range(peak_index - 1, -1, -1)
    else:
        segment_indexes = range(peak_index, len(current) - 1)

    for left_index in segment_indexes:
        right_index = left_index + 1
        left_offset = current[left_index] - half_height
        right_offset = current[right_index] - half_height

        if left_offset == 0:
            return float(potential[left_index])
        if right_offset == 0:
            return float(potential[right_index])
        if left_offset * right_offset < 0:
            current_delta = current[right_index] - current[left_index]
            if current_delta == 0:
                return None
            fraction = (half_height - current[left_index]) / current_delta
            crossing = potential[left_index] + fraction * (
                potential[right_index] - potential[left_index]
            )
            return float(crossing) if np.isfinite(crossing) else None

    return None


def peak_metrics(Alg_Potential, Alg_Current):
    """Return peak height/location/index and standard interpolated FWHM.

    FWHM is measured on the same Savitzky-Golay-smoothed, baseline-corrected
    peak used for peak height. A missing or ambiguous width is represented by
    ``None`` without discarding the other peak metrics.
    """
    Alg_Potential = np.asarray(Alg_Potential, dtype=float)
    Alg_Current = np.asarray(Alg_Current, dtype=float)
    if (
        Alg_Potential.ndim != 1
        or Alg_Current.ndim != 1
        or len(Alg_Potential) != len(Alg_Current)
        or len(Alg_Current) < 3
        or not np.all(np.isfinite(Alg_Potential))
        or not np.all(np.isfinite(Alg_Current))
    ):
        raise ValueError("peak region must contain at least 3 finite x/y samples")
    window = Change_Point_Detection.valid_savgol_window(len(Alg_Current), 5, 2)
    peak_smooth = savgol_filter(Alg_Current, window, 2)
    peak_index = int(np.argmax(peak_smooth))
    peak_height = float(peak_smooth[peak_index])
    peak_location = float(Alg_Potential[peak_index])

    peak_width = None
    if peak_height > 0 and 0 < peak_index < len(peak_smooth) - 1:
        half_height = peak_height / 2.0
        left_crossing = _half_height_crossing(
            Alg_Potential, peak_smooth, peak_index, half_height, -1
        )
        right_crossing = _half_height_crossing(
            Alg_Potential, peak_smooth, peak_index, half_height, 1
        )
        if left_crossing is not None and right_crossing is not None:
            candidate_width = abs(right_crossing - left_crossing)
            if np.isfinite(candidate_width) and candidate_width > 0:
                peak_width = float(candidate_width)

    return peak_height, peak_location, peak_index, peak_width


def peak_info(Alg_Potential, Alg_Current):
    """Return the legacy three peak values for backwards compatibility."""
    peak_height, peak_location, peak_index, _ = peak_metrics(
        Alg_Potential, Alg_Current
    )

    return peak_height, peak_location, peak_index


def process_file(args):
    Alg_File_Name, Alg_Data, Num_Curves,Alg_index,Alg_CPD_SM,Alg_CPD_CF,Alg_peak_region,Alg_noise_level,Fit_Alg    = args
    # print(Boundary)
    # x, y = read_csv_data(File_Name)
    CP_index = []
    CP_value = []
    Baseline_Mean = []
    Baseline_CI = []
    Peak_Mean = []
    Peak_Max = []
    Peak_Min = []
    Peak_Location = []
    Peak_Width = []
    for i in range( Num_Curves ):
        # print(Alg_File_Name,i,len(Alg_Data[0][i]),len(Alg_Data[1][i]))
        try:
            Curve_CP_index, Curve_CP_value,Alg_Current_CPD_smooth = Change_Point_Detection.CPD( Alg_Data[0][i],Alg_Data[1][i],Alg_CPD_SM,Alg_CPD_CF,Alg_peak_region,Alg_noise_level)
        except Exception as error:
            print(f"{Alg_File_Name} curve {i + 1} change-point detection failed: {error}")
            potential = np.asarray(Alg_Data[0][i]).reshape(-1)
            fallback = float(potential[0]) if len(potential) and np.isfinite(potential[0]) else 0
            CP_index.append((0, 0))
            CP_value.append((fallback, fallback))
            Baseline_Mean.append([])
            Baseline_CI.append([])
            Peak_Mean.append(0)
            Peak_Max.append(0)
            Peak_Min.append(0)
            Peak_Location.append(0)
            Peak_Width.append(None)
            continue
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
            Peak_Width.append(None)

            plt.figure(num = i+ 100)
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(16, 9))
            plt.plot(Alg_Data[0][i],Alg_Data[1][i], label='Raw_data', color='red')
            plt.plot(Alg_Data[0][i],Alg_Current_CPD_smooth, label='CPD smooth', color='black')
            plt.axvline(x=Curve_CP_value[1], color='black', label='Boundary-peak' )
            plt.axvline(x=Curve_CP_value[0],color='black')
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel('Current', fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results(Change point overlap)')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')
            continue

        Fit_Order = 3
        Num_Iter = 9999
        mask = np.ones(shape = len(Alg_Data[0][i]))
        mask[ int(Curve_CP_index[1]): int(Curve_CP_index[0])] = 0#be consist with boundary calculation
        weight = mask.astype(bool)

        #Fit_Alg = ['imodpoly4', 'penalized_poly4', 'pspline_derpsalsa', 'pspline_iarpls', 'pspline_iasls', 'pspline_mpls', 'fabc']

        Alg_baselines = []
        Fit_Alg_using = copy.copy(Fit_Alg)  #this list is to store the using algs
        plt.figure(num =   i+ 10000)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14
        plt.figure(figsize=(16, 9))
        plt.plot(Alg_Data[0][i],Alg_Data[1][i], label='Raw_data', color='red')
        plt.plot(Alg_Data[0][i],Alg_Current_CPD_smooth, label='CPD smooth', color='black')
        plt.axvline(x=Curve_CP_value[1], color='black', label='Boundary-peak' )
        plt.axvline(x=Curve_CP_value[0],color='black')
        index_baseline_fitting_standard = [] #index for baseline not satisfiled baseline_fitting_standard()
        for fitting_alg in Fit_Alg:
            try:
                (baseline, para), error = get_algo_instance(fitting_alg,Alg_Data[0][i],Alg_Current_CPD_smooth,Fit_Order,Num_Iter,weight)
            except Exception as error:
                print(f"{Alg_File_Name} curve {i + 1} {fitting_alg} failed: {error}")
                if fitting_alg in Fit_Alg_using:
                    Fit_Alg_using.remove(fitting_alg)
                continue
            if error:
                print(Alg_File_Name,Fit_Alg,error)

            elif baseline_fitting_standard(  Curve_CP_index, Alg_Current_CPD_smooth,baseline  ):
                # print(fitting_alg,'works')
                Alg_baselines.append(baseline)
                plt.plot(Alg_Data[0][i],baseline, label=fitting_alg)
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
            Peak_Width.append(None)
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel('Current', fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results(Fitting Failed)')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')
            continue

        try:
            if len(Alg_baselines) <5:  #no extreme value detection
                Curve_Baseline_Mean, Curve_Baseline_CI =get_CI(Alg_baselines)
            else:
                Curve_Baseline_Mean, Curve_Baseline_CI,Curve_Outlier_Info =  extreme_baseline_detection(Alg_baselines)
            Baseline_Mean.append(Curve_Baseline_Mean)
            Baseline_CI.append( Curve_Baseline_CI )
        except Exception as error:
            print(f"{Alg_File_Name} curve {i + 1} confidence calculation failed: {error}")
            Baseline_Mean.append([])
            Baseline_CI.append([])
            Peak_Mean.append(0)
            Peak_Max.append(0)
            Peak_Min.append(0)
            Peak_Location.append(0)
            Peak_Width.append(None)
            plt.close('all')
            continue
        # for j in range(len(Fit_Alg_using) ):
        #     if Curve_Outlier_Info[j]:
        #         plt.plot(Alg_Data[0][i],Alg_baselines[j], label=Fit_Alg_using[j])
        #     else:
        #         plt.plot(Alg_Data[0][i],Alg_baselines[j], '-.' , label=Fit_Alg_using[j])


        if Curve_Baseline_Mean:
            try:
                Curve_Peak_Mean,Curve_Peak_Location,Curve_Peak_Index,Curve_Peak_Width = peak_metrics( Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a - b for a, b in zip(Alg_Current_CPD_smooth, Curve_Baseline_Mean)][ int(Curve_CP_index[1]):int(Curve_CP_index[0])]  )
            except Exception as error:
                print(f"{Alg_File_Name} curve {i + 1} peak calculation failed: {error}")
                Peak_Mean.append(0)
                Peak_Max.append(0)
                Peak_Min.append(0)
                Peak_Location.append(0)
                Peak_Width.append(None)
                plt.close('all')
                continue
            Peak_Mean.append(Curve_Peak_Mean)
            Peak_Location.append(Curve_Peak_Location)
            Peak_Width.append(Curve_Peak_Width)

            Peak_Min.append(Curve_Peak_Mean - Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ] )
            Peak_Max.append(Curve_Peak_Mean +  Curve_Baseline_CI[ Curve_Peak_Index + int(Curve_CP_index[1])  ])
            plt.plot(Alg_Data[0][i],Curve_Baseline_Mean, label='baseline', color='yellow')
            plt.fill_between(
                Alg_Data[0][i],
                [a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                [a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                color='blue', alpha=0.2, label='99% Confidence Interval of Baseline'
            )
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel('Current', fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')



            plt.figure(num = i+ 1)
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(16, 9))
            plt.plot(Alg_Data[0][i],Alg_Data[1][i], label='Raw_data', color='red')
            plt.plot(Alg_Data[0][i],Baseline_Mean[i], label='Baseline', color='blue')
            plt.axvline(x=Curve_CP_value[1], color='red', label='Boundary-peak' )
            plt.axvline(x=Curve_CP_value[0],color='red')
            plt.fill_between(
                Alg_Data[0][i],
                [a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                [a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                color='blue', alpha=0.2, label='99% Confidence Interval of Baseline'
            )
            plt.plot(Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],[a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_Mean ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ], label='Peak', color='green')
            # plt.fill_between(
            #     Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     [a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_CI[0] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     [a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_CI[1] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     color='green', alpha=0.2, label='95% Confidence Interval of Peak'
            # )
            plt.scatter( Curve_Peak_Location,Curve_Peak_Mean,label= 'Peak Height' )
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel('Current', fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + '.png')

            # plt.figure(num =  Alg_File_Index + i+ 100) # back to draw alg figures


        else:
            print(Alg_File_Name, i,'Failed')
            Peak_Mean.append(0)
            Peak_Max.append(0)
            Peak_Min.append(0)
            Peak_Location.append(0)
            Peak_Width.append(None)
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel('Current', fontsize=14, fontname='Arial')# back to draw alg figures
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')

    return Alg_File_Name ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location, Peak_Width, Alg_index




def _report_analysis_progress(
    progress_callback, percent, message, completed_files=0, total_files=0
):
    """Report progress without allowing a GUI callback to break analysis."""
    if progress_callback is None:
        return
    try:
        progress_callback(percent, message, completed_files, total_files)
    except Exception as error:
        print(f"Could not report analysis progress: {error}")


def data_analysis(
    data_fromGUI,
    CPD_SM,
    CPD_CF,
    Alg_peak,
    Alg_noise_level,
    Alg_fitting_set,
    progress_callback=None,
):
        # print(result)
        freeze_support()
        #data_fromGUI = {'pssession': [{'file_type': 'pssession', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/3.pssession', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/2.pssession'], 'frequency': None, 'amplitude': None}], 'csv': [{'file_type': 'csv', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_30mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_50mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/40Hz_10mV.csv'], 'frequency': 0.0, 'amplitude': 0.0}]}

        if not os.path.exists('Fig_Saved'):
            os.makedirs('Fig_Saved')

        files = [[],[]]

        if 'pssession' in data_fromGUI:
            pssession_files = data_fromGUI['pssession']
            files[0].append(pssession_files['file_names'][:])
        if 'csv' in data_fromGUI:
            csv_files = data_fromGUI['csv']
            files[1].append(csv_files['file_names'][:])

        total_selected_files = sum(
            len(group[0]) for group in files if group and isinstance(group[0], list)
        )
        files_read = 0
        _report_analysis_progress(
            progress_callback,
            10,
            f"Reading {total_selected_files} selected file(s)...",
            0,
            total_selected_files,
        )

        data_save = {}
        num_curves = []
        raw_data = []#save C,V info, curves in one file is one sub list
        file_name = []
        curve_index = []

        if files[0] and len(files[0][0]) > 0:
            print('Reading Pssession File')
            for file in files[0][0]: #curves,times,len_curves,freq,amp,e_step
                try:
                    file_read_result = read_pssession_file(file)
                except Exception as error:
                    print(error)
                    continue
                finally:
                    files_read += 1
                    read_percent = 10 + (
                        15 * files_read / total_selected_files
                        if total_selected_files
                        else 15
                    )
                    _report_analysis_progress(
                        progress_callback,
                        read_percent,
                        f"Read {files_read} of {total_selected_files} selected file(s).",
                        0,
                        total_selected_files,
                    )
                file_name.append(file)
                data_save[file] = {}
                raw_data.append(file_read_result[0])
                a = file_read_result[2]#numger of curves
                # print(file,a,'curves')
                num_curves.append(a )

                for i in range(a):

                    data_save[file]['Curve No. '+str(i+1)] = {
                                    'Date and time measurement': file_read_result[1][i],
                                    'Channel ': file_read_result[6][i],
                                    'Frequence ': file_read_result[3][i],
                                    'Amplitude ': file_read_result[4][i],
                                    'E Step ': file_read_result[5][i],
                                    'Raw Poetntial ': file_read_result[0][0][i],
                                    'Raw Current': file_read_result[0][1][i],

                    }

        if files[1] and len(files[1][0]) > 0:
            print('Reading CSV File')
            for file in files[1][0]:
                try:
                    file_read_result = read_csv_file(file)
                except Exception as error:
                    print(f"Could not read CSV file {file}: {error}")
                    continue
                finally:
                    files_read += 1
                    read_percent = 10 + (
                        15 * files_read / total_selected_files
                        if total_selected_files
                        else 15
                    )
                    _report_analysis_progress(
                        progress_callback,
                        read_percent,
                        f"Read {files_read} of {total_selected_files} selected file(s).",
                        0,
                        total_selected_files,
                    )
                file_name.append(file)
                data_save[file] = {}
                raw_data.append(file_read_result[0])
                a = file_read_result[2]
                print(file,a,'curves')
                Time_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in  file_read_result[1]]
                num_curves.append(a )
                for i in range(a):
                    data_save[file]['Curve No. '+str(i+1)] = {
                                    'Date and time measurement': Time_str[i],
                                    'Raw Poetntial ': file_read_result[0][0][i],
                                    'Raw Current': file_read_result[0][1][i],
                    }

        args = [(file_name[i], raw_data[i],num_curves[i],i,CPD_SM,CPD_CF,Alg_peak,Alg_noise_level, Alg_fitting_set ) for i in range(len(raw_data))]

        if not args:
            raise ValueError("No valid input files were provided for analysis")
        _report_analysis_progress(
            progress_callback,
            25,
            f"Analyzing 0 of {len(args)} valid file(s)...",
            0,
            len(args),
        )
        num_cpus = max(1, min(len(args), cpu_count() - 2))

        print(f"Using {num_cpus} CPUs for Analysis" )

        with Pool(processes=num_cpus) as pool:
            #results = pool.map(process_file, args)

            results = []
            # start_time = time.time()
            for result in tqdm(pool.imap_unordered(process_file, args), total=len(args)):
                results.append(result)
                completed_files = len(results)
                _report_analysis_progress(
                    progress_callback,
                    25 + 65 * completed_files / len(args),
                    f"Analyzed {completed_files} of {len(args)} valid file(s).",
                    completed_files,
                    len(args),
                )
                # elapsed_time = time.time() - start_time
                # tqdm.write(f"Elapsed time: {elapsed_time:.2f} seconds")


        for result in results: #Alg_File_Index ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location
            file_name_save, cp_index, cp_value,baseline_mean,baseline_ci,peak_mean,peak_max,peak_min, peak_loc, peak_width, file_index =  result

            for curve_index in range(num_curves[file_index]):
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Change Point Indexes '] = cp_index[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Change Point Values '] = cp_value[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Baseline Mean '] = baseline_mean[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ '99\\% Confidence Interval of Baseline: '] = baseline_ci[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Peak Value '] =peak_mean[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ '99\\% Confidence Interval of Peak Value'] =[peak_min[curve_index] ,peak_max[curve_index] ]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Peak Location: '] = peak_loc[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][PEAK_WIDTH_KEY] = peak_width[curve_index]
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'Concentration'] = "undefined"
                data_save[ file_name_save]['Curve No. '+str(curve_index+1)][ 'review_status'] = "pass" if peak_mean[curve_index] else "fail"


        #analysis_results = dict(sorted(analysis_results.items(), key=lambda item: item[1]['File No. ']))

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # save_filename = f"data_{current_time}.json"

        write_json(RESULTS_PATH, data_save)
        gc.collect()

        return data_save



def Time_Series_Analysis(Alg_names,Alg_num_curves,Data_Save_IQR) :



    Num_Files = len(Alg_names)

    ave_peak = np.zeros(shape = (Alg_num_curves, Num_Files ) )
    for j in range(Alg_num_curves):

        Peak_median = []

        Peak_CI_IQR = []

        Time = []
        for i in range(Num_Files):

            #basepeak result
            Time.append( Data_Save_IQR[Alg_names[i] ]['Curve No. '+str(j+1 )]['Date and time measurement'] )

            Peak_median.append( Data_Save_IQR[Alg_names[i] ]['Curve No. '+str(j+1)]['Peak Value '] )

            curve_data = Data_Save_IQR[Alg_names[i]]["Curve No. " + str(j + 1)]
            Peak_CI_IQR.append(
                curve_data.get(
                    "99\\% Confidence Interval of Peak Value",
                    curve_data.get("95\\% Confidence Interval of Peak Value", [0, 0]),
                )
            )

        Time = [parser.parse(t) for t in Time]


        combined_list = list(zip(Time, Peak_median, Peak_CI_IQR))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        Sorted_Time, Sorted_Peak_median, Sorted_Peak_CI_IQR= zip(*sorted_combined_list)
        Sorted_Time = np.array(Sorted_Time)

        Sorted_Peak_median = np.array(Sorted_Peak_median)

        Sorted_Peak_CI_IQR = np.array(Sorted_Peak_CI_IQR)

        ave_peak[j,:] =  Sorted_Peak_median / np.max(Sorted_Peak_median)




        degree = 10
        # base_time = Sorted_Time[0:].min()
        time_in_hours =  [(t - Sorted_Time[5]).total_seconds() / 3600 for t in Sorted_Time[0:]]



        fig = go.Figure()

        # 添加中位数折线图
        fig.add_trace(go.Scatter(
            x=time_in_hours[5:],
            y=Sorted_Peak_median[5:],
            mode='lines',
            line=dict(color='black', width=0.75),
            name='Median'
        ))

        # 添加置信区间
        fig.add_trace(go.Scatter(
            x=time_in_hours[5:],
            y=Sorted_Peak_CI_IQR[5:, 1],
            fill=None,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.2)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=time_in_hours[5:],
            y=Sorted_Peak_CI_IQR[5:, 0],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.2)'),
            name='99% Confidence Interval'
        ))

        # 添加散点图
        fig.add_trace(go.Scatter(
            x=time_in_hours[5:],
            y=Sorted_Peak_median[5:],
            mode='markers',
            marker=dict(size=6),
            name='Median Points'
        ))

        # 添加标题和标签
        fig.update_layout(
            title='Peak Height with Time',
            xaxis_title='Time (hours)',
            yaxis_title='Peak Height',
            legend=dict(x=0, y=1.0),
        )

        # 导出为 HTML
        fig.write_html('Time_Series_Curve_No_' + str(j+1) + '_testfilter.html')



        Results2Save = {}
        Time_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in Sorted_Time]
        Sorted_Peak_median = Sorted_Peak_median.tolist()
        Sorted_Peak_CI_IQR = Sorted_Peak_CI_IQR.tolist()
        for index2save in range(len(Sorted_Time)):
            Results2Save[index2save] = {}
            Results2Save[index2save]['Time'] = Time_str[index2save]
            Results2Save[index2save]['Peak Height'] = Sorted_Peak_median[index2save]
            Results2Save[index2save]['Confidence Interval'] = Sorted_Peak_CI_IQR[index2save]
        with open('TS_Curve No.'+ str(j+1) + '.json', 'w') as json_file:
            json.dump(Results2Save, json_file,indent=4,  ensure_ascii=False)

    mean_ave_peak = np.zeros(Num_Files)
    standard_error = np.zeros(Num_Files)
    for i in range(Num_Files):
        non_zero_data = []
        for j in range(Alg_num_curves):
            if ave_peak[j][i] != 0 and not np.isnan(ave_peak[j][i]):
                non_zero_data.append(ave_peak[j][i])
        non_zero_data = np.array(non_zero_data)
        if len(non_zero_data) > 0:
            mean_ave_peak[i] = np.mean(non_zero_data)
            # 计算标准误差作为误差条（标准差除以有效样本数的平方根）
            standard_error[i] = np.std(non_zero_data) / np.sqrt(len(non_zero_data))
        else:
            mean_ave_peak[i] = 0  # 如果所有值都为0，平均值设为0，误差条为0
            standard_error[i] = 0

    # mean_ave_peak = np.mean(ave_peak, axis=0)

    # # 2. 计算标准偏差，沿 Alg_num_curves 方向 (第 0 维) 计算
    # std_dev = np.std(ave_peak, axis=0)

    # # 3. 计算标准误差（SE = 标准偏差 / 样本数量的平方根）
    # standard_error = std_dev / np.sqrt(Alg_num_curves)

    fig = go.Figure()

    # 添加均值和误差条
    fig.add_trace(go.Scatter(
        x=Sorted_Time,
        y=mean_ave_peak,
        mode='markers+lines',  # 绘制点和线
        name='Signal',
        error_y=dict(
            type='data',  # 使用数据作为误差条
            array=standard_error,  # 误差条的值
            visible=True,  # 显示误差条
            color='red'    # 误差条颜色
        )
    ))

    # 3. 设置图表标题和轴标签
    fig.update_layout(
        title="Mean Values with Error Bars",
        xaxis_title="Time",
        yaxis_title="Normalized Signal",
        showlegend=True
    )
    fig.write_html('Time_Series_ave.html')

    ave_result2save = {}
    mean_ave_peak = list(mean_ave_peak)
    for i in range(Num_Files):
        ave_result2save[str(i+1)] = {}
        ave_result2save[str(i+1)]['Time'] = Time_str[i]
        ave_result2save[str(i+1)]['Normalized Peak Hight'] = mean_ave_peak[i]
        ave_result2save[str(i+1)]['Error Bar'] =standard_error[i]

    with open('TS_Curve_Average.json', 'w') as json_file:
        json.dump(ave_result2save, json_file,indent=4,  ensure_ascii=False)






# if __name__ == '__main__':
# #     # a =   {'pssession': [{'file_type': 'pssession', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/3.pssession', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/2.pssession'], 'frequency': None, 'amplitude': None}], 'csv': [{'file_type': 'csv', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_30mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_50mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/40Hz_10mV.csv'], 'frequency': 0.0, 'amplitude': 0.0}]}
#     # os.chdir('data4compare')
#     a = {
#     "pssession": {
#         "file_type": "pssession",
#         "file_names": [
#             'test.pssession'
#         ]
#     }
# }
#     SR_weight = 0.5

#     with open('Algorithm Setting.json', 'r', encoding='utf-8') as file:
#         data_algs = json.load(file)
#     name = str(int(SR_weight/0.01))


#     # names = list(data_algs.keys())
#     # for alg_index in names:
#     #     if data_algs[alg_index]["Success Weight"] == round(SR_weight,2):
#     #         break
#     noise_level = 3

#     data_analysis(a,
#                   data_algs[name][ "CPD Search Model"],
#                   data_algs[name][ "CPD Cost Function"],
#                   float(data_algs[name][ "Ratio For Peak"]),
#                   noise_level,
#                   list(ast.literal_eval(data_algs[name][ "Baseline Fitting Algorithms"]) ))
#     print('end')
#     a = {
#     "csv": {
#         "file_type": "csv",
#         "file_names": [

#         ]
#     }
# }

#     os.chdir('training set')

#     folder_list = os.listdir()
#     #folder_list = ['20240802_in vivo_Day 4']
#     for folder_index in folder_list:
#         # if folder_index != '01_npAu_6o':
#         #     continue
#         os.chdir(folder_index)


#         folder_path = './Fig_Saved'
#         if os.path.exists(folder_path):

#             shutil.rmtree(folder_path)

#         os.makedirs(folder_path,exist_ok=True)
#         # print('y')
#         file_list  = []

#         #exclude path.txt
#         for file_in_folder in os.listdir():
#             #if file_in_folder .endswith('.psession'):
#             if file_in_folder .endswith('.csv'):
#                 file_list.append(file_in_folder )
#         #a['pssession']['file_names'] = file_list
#         a['csv']['file_names'] = file_list
#         #data_analysis(a)


#         with open('results.json', 'r', encoding='utf-8') as file:
#             data_IQR = json.load(file)
#         names = list(data_IQR.keys())
#         num_curves = len(  list(data_IQR[names[0]].keys() ))

#         Time_Series_Analysis(names,num_curves,data_IQR)
#         os.chdir('../')
