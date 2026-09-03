import pspython.pspyfiles as pspyfiles
import Change_Point_Detection
from ruptures.exceptions import BadSegmentationParameters
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
from Algs import MULTI_PEAK_BASELINE_ALGORITHMS, get_algo_instance
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
from cv_processing import (
    CVProcessingError,
    CVSignal,
    derived_branch_filename,
    load_cv_branches,
    split_cv_signal,
)


CSV_HEADER_MARKER = "Date and time measurement:"
CSV_ENCODING_SAMPLE_SIZE = 64 * 1024

MULTI_PEAK_FIGURE_ROOT = os.path.join(
    "Fig_Saved", "Shared_Outer_Wing_Baselines"
)
MULTI_PEAK_MAX_ITERATIONS = 250

CV_SCAN_DIRECTION_KEY = "CV Scan Direction"
CV_CURRENT_SIGN_KEY = "Current Sign Multiplier"
CV_ORIGINAL_CURRENT_KEY = "Original Raw Current"
CV_ORIGINAL_SOURCE_KEY = "Original Source File Path"
CV_CURRENT_UNIT_KEY = "Current Unit"
SIGNED_PEAK_CURRENT_KEY = "Signed Peak Current"
DEFAULT_BASELINE_SCREEN_TOLERANCE = 0.10
CV_BASELINE_SCREEN_TOLERANCE = 0.12


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


def read_csv_file(filename, *, trim_edges=True):
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
    if trim_edges and len(result) <= 41:
        raise ValueError("CSV file must contain more than 41 data rows.")
    #print(result)
    curves = [[[] for _ in range(len_curves)],[[] for _ in range(len_curves)]]

    curve_data = result[20:-21] if trim_edges else result
    for i in range(len_curves):
        for row in curve_data:
            curves[0][i].append(float(row[i * 2]))
            curves[1][i].append(float(row[1 + i * 2]) )


    return(curves,parsed_dates, len_curves)

# SDK only

def read_pssession_file(filename, *, trim_edges=True):
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

    sample_slice = slice(20, -20) if trim_edges else slice(None)
    curves[0] = [measurement.potential_arrays[0][sample_slice] for measurement in Measurements]
    curves[1] = [measurement.current_arrays[0][sample_slice] for measurement in Measurements]
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


def baseline_fitting_standard(
    Alg_Boundarys,
    Alg_Raw_Current,
    Alg_Baseline_Current,
    Alg_Excluded_Boundaries=None,
    *,
    max_above_fraction=DEFAULT_BASELINE_SCREEN_TOLERANCE,
    max_mwse=DEFAULT_BASELINE_SCREEN_TOLERANCE,
):
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
    try:
        max_above_fraction = float(max_above_fraction)
        max_mwse = float(max_mwse)
    except (TypeError, ValueError):
        return False
    if not (
        np.isfinite(max_above_fraction)
        and np.isfinite(max_mwse)
        and 0 <= max_above_fraction <= 1
        and max_mwse >= 0
    ):
        return False
    higher_counter = np.sum(Alg_Baseline_Current[Alg_Boundarys[1]:Alg_Boundarys[0]+1] > [ a*1.00 for a in Alg_Raw_Current[Alg_Boundarys[1]:Alg_Boundarys[0]+1]])
    if higher_counter > (Alg_Boundarys[0] - Alg_Boundarys[1]) * max_above_fraction:
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
    if Alg_Excluded_Boundaries is None:
        Alg_raw = np.concatenate(( Alg_Raw_Current[ :Alg_Boundarys[1] ], Alg_Raw_Current[Alg_Boundarys[0]: ]))
        Alg_baseline = np.concatenate(( Alg_Baseline_Current[ :Alg_Boundarys[1] ], Alg_Baseline_Current[Alg_Boundarys[0]: ]))
        source_indexes = None
        excluded_boundaries = [Alg_Boundarys]
    else:
        excluded_boundaries = []
        included = np.ones(len(Alg_Raw_Current), dtype=bool)
        try:
            for boundary in Alg_Excluded_Boundaries:
                excluded_upper, excluded_lower = (
                    int(boundary[0]),
                    int(boundary[1]),
                )
                if not 0 <= excluded_lower < excluded_upper < len(
                    Alg_Raw_Current
                ):
                    return False
                excluded_boundaries.append(
                    (excluded_upper, excluded_lower)
                )
                included[excluded_lower:excluded_upper] = False
        except (TypeError, ValueError, IndexError):
            return False
        if not excluded_boundaries:
            return False
        source_indexes = np.flatnonzero(included)
        Alg_raw = Alg_Raw_Current[included]
        Alg_baseline = Alg_Baseline_Current[included]
    if len(Alg_raw) == 0:
        return False
    signal_range = np.ptp(Alg_raw)
    if not np.isfinite(signal_range) or signal_range == 0:
        return False
    sigma_2 = ( 0.25*(len(Alg_raw)) ) **2
    weights = []
    Square_Error = [] # (y-y*)**2
    if source_indexes is None:
        mid_point  = len(  Alg_Raw_Current[ :Alg_Boundarys[1] ])
        for point_index in range( len(Alg_raw)  ):
            if point_index < mid_point :
                Square_Error.append( (Alg_raw[point_index] - Alg_baseline[point_index]) **2  )
                weights.append( math.exp( - ( point_index  - mid_point-1)**2/sigma_2   )   )
            else:
                Square_Error.append( (Alg_raw[point_index] - Alg_baseline[point_index] )**2  )
                weights.append( math.exp( - ( point_index  - (mid_point) )**2/sigma_2   )   )
    else:
        boundary_edges = [
            edge
            for excluded_upper, excluded_lower in excluded_boundaries
            for edge in (max(0, excluded_lower - 1), excluded_upper)
        ]
        for point_index, source_index in enumerate(source_indexes):
            Square_Error.append(
                (Alg_raw[point_index] - Alg_baseline[point_index]) ** 2
            )
            distance = min(
                abs(int(source_index) - edge) for edge in boundary_edges
            )
            weights.append(math.exp(-(distance ** 2) / sigma_2))
    sum_weights = sum(weights)
    if sum_weights == 0:
        return False

    MWSE =  0

    for point_index in range( len(Alg_raw)  ):
        MWSE += weights[point_index] * Square_Error[point_index]

    MWSE = MWSE / sum_weights / (signal_range ** 2)
    if MWSE > max_mwse:
        print('MWSE: ', MWSE)
    #print(pearson_r,p_value)
    return MWSE < max_mwse


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
DETECTED_CP_INDEXES_KEY = "Detected Change Point Indexes "
DETECTED_CP_VALUES_KEY = "Detected Change Point Values "
SOURCE_FILE_KEY = "Source File Path"
PEAK_NUMBER_KEY = "Peak Number"
PEAK_LABEL_KEY = "Peak Label"

_PEAK_ORDINALS = (
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
    "Eleventh",
    "Twelfth",
    "Thirteenth",
    "Fourteenth",
    "Fifteenth",
    "Sixteenth",
    "Seventeenth",
    "Eighteenth",
    "Nineteenth",
    "Twentieth",
)


def peak_label(peak_index):
    """Return a stable, human-readable label for a zero-based peak index."""

    if isinstance(peak_index, bool) or not isinstance(
        peak_index, (int, np.integer)
    ):
        raise ValueError("peak index must be a non-negative integer")
    peak_index = int(peak_index)
    if peak_index < 0:
        raise ValueError("peak index must be a non-negative integer")
    if peak_index < len(_PEAK_ORDINALS):
        return _PEAK_ORDINALS[peak_index]

    number = peak_index + 1
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def peak_result_name(file_name, peak_index, peak_count):
    """Return the logical result key for one peak in a source file."""

    if peak_count == 1:
        return str(file_name)
    return f"{file_name}-{peak_label(peak_index)}"


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
    if len(args) not in (9, 10, 11):
        raise ValueError("single-peak worker expects 9, 10 or 11 arguments")
    Alg_File_Name, Alg_Data, Num_Curves,Alg_index,Alg_CPD_SM,Alg_CPD_CF,Alg_peak_region,Alg_noise_level,Fit_Alg = args[:9]
    measurement_type = _normalise_measurement_type(args[9] if len(args) >= 10 else "swv")
    plot_sign = args[10] if len(args) == 11 and measurement_type == "cv" else 1
    if plot_sign not in (-1, 1):
        raise ValueError("CV current sign must be -1 or 1")

    def display_current(values):
        return np.asarray(values, dtype=float) * plot_sign

    current_label = "Current (µA)" if measurement_type == "cv" else "Current"
    # Diffusion tails in CV scans are more asymmetric than the SWV signals
    # this screen was tuned for.  A local 12% tolerance keeps the same CPD and
    # baseline ensemble while accepting the four validated ferri/ferrocyanide
    # branches; the established SWV threshold remains unchanged.
    baseline_tolerance = (
        CV_BASELINE_SCREEN_TOLERANCE
        if measurement_type == "cv"
        else DEFAULT_BASELINE_SCREEN_TOLERANCE
    )
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
            plt.plot(Alg_Data[0][i],display_current(Alg_Data[1][i]), label='Raw_data', color='red')
            plt.plot(Alg_Data[0][i],display_current(Alg_Current_CPD_smooth), label='CPD smooth', color='black')
            plt.axvline(x=Curve_CP_value[1], color='black', label='Boundary-peak' )
            plt.axvline(x=Curve_CP_value[0],color='black')
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel(current_label, fontsize=14, fontname='Arial')
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
        plt.plot(Alg_Data[0][i],display_current(Alg_Data[1][i]), label='Raw_data', color='red')
        plt.plot(Alg_Data[0][i],display_current(Alg_Current_CPD_smooth), label='CPD smooth', color='black')
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

            elif (
                baseline_fitting_standard(
                    Curve_CP_index,
                    Alg_Current_CPD_smooth,
                    baseline,
                    max_above_fraction=baseline_tolerance,
                    max_mwse=baseline_tolerance,
                )
                if measurement_type == "cv"
                else baseline_fitting_standard(
                    Curve_CP_index,
                    Alg_Current_CPD_smooth,
                    baseline,
                )
            ):
                # print(fitting_alg,'works')
                Alg_baselines.append(baseline)
                plt.plot(Alg_Data[0][i],display_current(baseline), label=fitting_alg)
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
            plt.ylabel(current_label, fontsize=14, fontname='Arial')
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
            plt.plot(Alg_Data[0][i],display_current(Curve_Baseline_Mean), label='baseline', color='yellow')
            plt.fill_between(
                Alg_Data[0][i],
                [plot_sign * a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                [plot_sign * a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                color='blue', alpha=0.2, label='99% Confidence Interval of Baseline'
            )
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel(current_label, fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')



            plt.figure(num = i+ 1)
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14
            plt.figure(figsize=(16, 9))
            plt.plot(Alg_Data[0][i],display_current(Alg_Data[1][i]), label='Raw_data', color='red')
            plt.plot(Alg_Data[0][i],display_current(Baseline_Mean[i]), label='Baseline', color='blue')
            plt.axvline(x=Curve_CP_value[1], color='red', label='Boundary-peak' )
            plt.axvline(x=Curve_CP_value[0],color='red')
            plt.fill_between(
                Alg_Data[0][i],
                [plot_sign * a - b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                [plot_sign * a + b for a, b in zip(Curve_Baseline_Mean, Curve_Baseline_CI)],
                color='blue', alpha=0.2, label='99% Confidence Interval of Baseline'
            )
            plt.plot(Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],display_current([a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_Mean ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ]), label='Peak', color='green')
            # plt.fill_between(
            #     Alg_Data[0][i][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     [a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_CI[0] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     [a-b for a,b in zip(Alg_Data[1][i],Curve_Baseline_CI[1] ) ][ int(Curve_CP_index[1]):int(Curve_CP_index[0]) ],
            #     color='green', alpha=0.2, label='95% Confidence Interval of Peak'
            # )
            plt.scatter( Curve_Peak_Location,plot_sign * Curve_Peak_Mean,label= 'Peak Height' )
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel(current_label, fontsize=14, fontname='Arial')
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + '.png')
            plt.close('all')

            # plt.figure(num =  Alg_File_Index + i+ 100) # back to draw alg figures


        else:
            print(Alg_File_Name, i,'Failed')
            Peak_Mean.append(0)
            Peak_Max.append(0)
            Peak_Min.append(0)
            Peak_Location.append(0)
            Peak_Width.append(None)
            plt.xlabel('Potential', fontsize=14, fontname='Arial')
            plt.ylabel(current_label, fontsize=14, fontname='Arial')# back to draw alg figures
            plt.legend()
            plt.title('Data Analysis Results')
            plt.savefig('Fig_Saved/'+ os.path.basename(Alg_File_Name) + '_'+str(i) + 'alg.png')
            plt.close('all')

    return Alg_File_Name ,CP_index, CP_value, Baseline_Mean, Baseline_CI, Peak_Mean, Peak_Max, Peak_Min, Peak_Location, Peak_Width, Alg_index


def _multi_peak_failure(region, potential):
    """Build a complete failed-peak result while retaining detected CPs."""

    fallback = (
        float(potential[0])
        if len(potential) and np.isfinite(potential[0])
        else 0.0
    )
    if isinstance(region, dict):
        boundary_indexes = region.get("boundary_indexes", (0, 0))
        boundary_values = region.get("boundary_values", (fallback, fallback))
        detected_indexes = region.get("change_point_indexes", ())
        detected_values = region.get("change_point_values", ())
    else:
        boundary_indexes = (0, 0)
        boundary_values = (fallback, fallback)
        detected_indexes = ()
        detected_values = ()
    return {
        "Change Point Indexes ": [int(value) for value in boundary_indexes],
        "Change Point Values ": [float(value) for value in boundary_values],
        DETECTED_CP_INDEXES_KEY: [int(value) for value in detected_indexes],
        DETECTED_CP_VALUES_KEY: [float(value) for value in detected_values],
        "Change Point Source": "automatic",
        "Baseline Mean ": [],
        "99\\% Confidence Interval of Baseline: ": [],
        "Peak Value ": 0,
        "99\\% Confidence Interval of Peak Value": [0, 0],
        "Peak Location: ": 0,
        PEAK_WIDTH_KEY: None,
        "review_status": "fail",
    }


def _outer_wing_baseline_mask(regions, sample_count):
    """Keep only samples outside the complete detected multi-peak span.

    Baseline algorithms receive the left wing before the earliest peak and the
    right wing after the latest peak.  Inter-peak samples are deliberately not
    fitting data.  ``excluded_boundaries`` still contains each individual peak
    so screening can evaluate the non-peak left, middle, and right regions.
    """

    if isinstance(sample_count, bool) or not isinstance(
        sample_count, (int, np.integer)
    ) or sample_count < 1:
        raise ValueError("sample count must be a positive integer")

    intervals = []
    excluded_boundaries = []
    for region in regions:
        try:
            indexes = tuple(int(value) for value in region["change_point_indexes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every peak region must contain numeric change-point indexes"
            ) from exc
        if len(indexes) < 2:
            raise ValueError(
                "every peak region must contain at least two change points"
            )
        lower_index = min(indexes)
        upper_index = max(indexes)
        if not 0 <= lower_index < upper_index < sample_count:
            raise ValueError("peak region change-point indexes are out of range")
        intervals.append((lower_index, upper_index))
        excluded_boundaries.append((upper_index, lower_index))

    if len(intervals) < 2:
        raise ValueError("multi-peak baseline fitting requires at least two regions")

    ordered_intervals = sorted(intervals)
    for previous, current in zip(ordered_intervals, ordered_intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError("detected peak regions must not overlap")

    left_end = ordered_intervals[0][0]
    right_start = ordered_intervals[-1][1]
    if left_end == 0 or right_start >= sample_count - 1:
        raise ValueError("both outer baseline wings must contain samples")

    mask = np.zeros(int(sample_count), dtype=bool)
    mask[:left_end] = True
    mask[right_start + 1 :] = True
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two outer-wing samples are required")
    return mask, excluded_boundaries


def _baseline_only_signal(potential, smoothed, fitting_mask):
    """Replace detected peaks using only interpolation from baseline regions.

    Several pybaselines algorithms treat supplied weights as initial weights
    and may assign non-zero weights to a masked point in later iterations.  By
    inpainting every excluded sample first, even those algorithms can only see
    measured values from the caller's retained baseline mask. Multi-peak shared
    fitting supplies only the two outer wings; the single-peak caller can use
    its own retained regions.
    """

    potential = np.asarray(potential, dtype=float)
    smoothed = np.asarray(smoothed, dtype=float)
    fitting_mask = np.asarray(fitting_mask, dtype=bool)
    if not (
        potential.ndim == smoothed.ndim == fitting_mask.ndim == 1
        and len(potential) == len(smoothed) == len(fitting_mask)
        and len(potential) > 1
    ):
        raise ValueError("baseline fitting arrays must be one-dimensional and equal length")
    if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(smoothed)):
        raise ValueError("baseline fitting arrays must contain only finite values")
    if np.count_nonzero(fitting_mask) < 2:
        raise ValueError("at least two baseline-region samples are required")

    anchor_x = potential[fitting_mask]
    anchor_y = smoothed[fitting_mask]
    order = np.argsort(anchor_x)
    anchor_x = anchor_x[order]
    anchor_y = anchor_y[order]
    if np.any(np.diff(anchor_x) == 0):
        raise ValueError("baseline-region potential values must be unique")

    baseline_input = smoothed.copy()
    baseline_input[~fitting_mask] = np.interp(
        potential[~fitting_mask], anchor_x, anchor_y
    )
    return baseline_input


def _fit_shared_multi_peak_baseline(
    potential,
    smoothed,
    regions,
    fitting_algorithms,
    algorithm_runner=None,
    screen_function=None,
):
    """Fit once from the outer wings and intersect every peak screen."""

    if algorithm_runner is None:
        algorithm_runner = get_algo_instance
    if screen_function is None:
        screen_function = baseline_fitting_standard

    potential = np.asarray(potential, dtype=float)
    smoothed = np.asarray(smoothed, dtype=float)
    regions = list(regions)
    if (
        potential.ndim != 1
        or smoothed.ndim != 1
        or len(potential) != len(smoothed)
        or len(potential) < 2
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(smoothed))
    ):
        raise ValueError(
            "shared baseline arrays must be finite, one-dimensional, and equal length"
        )
    if len(regions) < 2 or not all(
        isinstance(region, dict) and region.get("valid") for region in regions
    ):
        raise ValueError("every requested multi-peak region must be valid")

    mask, excluded_boundaries = _outer_wing_baseline_mask(
        regions, len(potential)
    )
    baseline_input = _baseline_only_signal(potential, smoothed, mask)
    accepted_baselines = []
    for fitting_algorithm in fitting_algorithms:
        try:
            (baseline, _), error = algorithm_runner(
                fitting_algorithm,
                potential,
                baseline_input,
                3,
                MULTI_PEAK_MAX_ITERATIONS,
                mask,
            )
            baseline = np.asarray(baseline, dtype=float)
        except Exception:
            continue
        if (
            error
            or baseline.shape != smoothed.shape
            or not np.all(np.isfinite(baseline))
        ):
            continue

        screen_results = [
            screen_function(
                region["boundary_indexes"],
                smoothed,
                baseline,
                excluded_boundaries,
            )
            for region in regions
        ]
        if all(screen_results):
            accepted_baselines.append(baseline)

    if not accepted_baselines:
        raise ValueError(
            "no baseline algorithm was accepted by every peak screen"
        )

    if len(accepted_baselines) < 5:
        baseline_mean, baseline_half_width = get_CI(accepted_baselines)
    else:
        baseline_mean, baseline_half_width, _ = extreme_baseline_detection(
            accepted_baselines
        )
    baseline_mean = np.asarray(baseline_mean, dtype=float)
    baseline_half_width = np.asarray(baseline_half_width, dtype=float)
    if (
        baseline_mean.shape != smoothed.shape
        or baseline_half_width.shape != smoothed.shape
        or not np.all(np.isfinite(baseline_mean))
        or not np.all(np.isfinite(baseline_half_width))
    ):
        raise ValueError("shared baseline result does not match the source curve")
    return mask, accepted_baselines, baseline_mean, baseline_half_width


def _shared_multi_peak_result_updates(
    potential,
    smoothed,
    regions,
    baseline_mean,
    baseline_half_width,
):
    """Calculate every peak against one already-screened shared baseline."""

    potential = np.asarray(potential, dtype=float)
    smoothed = np.asarray(smoothed, dtype=float)
    baseline_mean = np.asarray(baseline_mean, dtype=float)
    baseline_half_width = np.asarray(baseline_half_width, dtype=float)
    regions = list(regions)
    if not (
        potential.ndim
        == smoothed.ndim
        == baseline_mean.ndim
        == baseline_half_width.ndim
        == 1
        and len(potential)
        == len(smoothed)
        == len(baseline_mean)
        == len(baseline_half_width)
    ):
        raise ValueError("shared peak result arrays must be one-dimensional and equal length")

    corrected = smoothed - baseline_mean
    region_order = sorted(
        range(len(regions)),
        key=lambda index: min(regions[index]["change_point_indexes"]),
    )
    calculated_results = []
    for peak_index, region in enumerate(regions):
        upper_index, lower_index = region["boundary_indexes"]
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        if not 0 <= lower_index < upper_index < len(potential):
            raise ValueError("peak region boundaries are out of range")

        peak_mean, peak_location, relative_peak_index, peak_width = peak_metrics(
            potential[lower_index:upper_index],
            corrected[lower_index:upper_index],
        )
        absolute_peak_index = lower_index + int(relative_peak_index)
        half_width = float(baseline_half_width[absolute_peak_index])
        peak_mean = float(peak_mean)

        if peak_width is None:
            region_position = region_order.index(peak_index)
            if region_position == 0:
                width_lower = 0
            else:
                previous_region = regions[region_order[region_position - 1]]
                previous_upper = max(
                    int(value)
                    for value in previous_region["change_point_indexes"]
                )
                width_lower = (previous_upper + lower_index) // 2
            if region_position == len(regions) - 1:
                width_upper = len(potential)
            else:
                next_region = regions[region_order[region_position + 1]]
                next_lower = min(
                    int(value) for value in next_region["change_point_indexes"]
                )
                width_upper = (upper_index + next_lower) // 2
            if width_upper - width_lower >= 3:
                _, _, extended_relative_index, extended_width = peak_metrics(
                    potential[width_lower:width_upper],
                    corrected[width_lower:width_upper],
                )
                extended_peak_index = width_lower + int(extended_relative_index)
                if lower_index <= extended_peak_index < upper_index:
                    peak_width = extended_width

        if not (
            np.isfinite(peak_mean)
            and np.isfinite(peak_location)
            and np.isfinite(half_width)
        ):
            raise ValueError("peak metrics contain non-finite values")
        calculated_results.append(
            {
                "Baseline Mean ": baseline_mean.tolist(),
                "99\\% Confidence Interval of Baseline: ": (
                    baseline_half_width.tolist()
                ),
                "Peak Value ": peak_mean,
                "99\\% Confidence Interval of Peak Value": [
                    peak_mean - half_width,
                    peak_mean + half_width,
                ],
                "Peak Location: ": float(peak_location),
                PEAK_WIDTH_KEY: peak_width,
                "review_status": "pass" if peak_mean else "fail",
            }
        )
    return calculated_results


def _multi_peak_figure_path(
    file_name, file_index, curve_index, peak_index, output_directory=None
):
    """Return a collision-resistant path for one shared-baseline peak view."""

    output_directory = output_directory or MULTI_PEAK_FIGURE_ROOT
    base_name = os.path.splitext(os.path.basename(str(file_name)))[0]
    safe_name = re.sub(r"[^\w.-]+", "_", base_name, flags=re.UNICODE).strip("._")
    if not safe_name:
        safe_name = "data"
    return os.path.join(
        output_directory,
        (
            f"{safe_name}_File_{int(file_index) + 1:03d}_"
            f"Curve_{int(curve_index) + 1:03d}_{peak_label(peak_index)}.png"
        ),
    )


def _save_multi_peak_figure(
    file_name,
    file_index,
    curve_index,
    peak_index,
    potential,
    current,
    smoothed,
    region,
    fitting_mask,
    baselines=(),
    baseline_mean=None,
    baseline_half_width=None,
    output_directory=None,
    review_status=None,
):
    """Save one shared-baseline peak view without breaking analysis."""

    try:
        potential = np.asarray(potential, dtype=float)
        current = np.asarray(current, dtype=float)
        smoothed = np.asarray(smoothed, dtype=float)
        fitting_mask = np.asarray(fitting_mask, dtype=bool)
        if not (
            potential.ndim == current.ndim == smoothed.ndim == fitting_mask.ndim == 1
            and len(potential) == len(current) == len(smoothed) == len(fitting_mask)
        ):
            raise ValueError("figure arrays must be one-dimensional and equal length")

        output_directory = output_directory or MULTI_PEAK_FIGURE_ROOT
        os.makedirs(output_directory, exist_ok=True)
        figure, axis = plt.subplots(figsize=(13, 8))
        axis.plot(potential, current, color="red", alpha=0.45, label="Raw current")
        axis.plot(potential, smoothed, color="black", linewidth=1.4, label="Smoothed current")
        axis.scatter(
            potential[fitting_mask],
            smoothed[fitting_mask],
            color="tab:green",
            s=10,
            alpha=0.7,
            label="Outer-wing fitting regions",
            zorder=3,
        )

        for baseline_index, baseline in enumerate(baselines):
            candidate = np.asarray(baseline, dtype=float)
            if candidate.shape == smoothed.shape:
                axis.plot(
                    potential,
                    candidate,
                    color="0.65",
                    linewidth=0.8,
                    alpha=0.45,
                    label="Accepted candidates" if baseline_index == 0 else None,
                )

        has_result = baseline_mean is not None
        if has_result:
            baseline_mean = np.asarray(baseline_mean, dtype=float)
            if baseline_mean.shape != smoothed.shape:
                raise ValueError("baseline mean does not match the source curve")
            axis.plot(
                potential,
                baseline_mean,
                color="tab:blue",
                linewidth=2.2,
                label="Shared baseline median",
            )
            if baseline_half_width is not None:
                half_width = np.asarray(baseline_half_width, dtype=float)
                if half_width.shape == smoothed.shape:
                    axis.fill_between(
                        potential,
                        baseline_mean - half_width,
                        baseline_mean + half_width,
                        color="tab:blue",
                        alpha=0.14,
                        label="Baseline algorithm spread",
                    )

        if isinstance(region, dict):
            boundary_values = region.get("boundary_values", ())
            if isinstance(boundary_values, (list, tuple)) and len(boundary_values) >= 2:
                for boundary_index, boundary in enumerate(boundary_values[:2]):
                    axis.axvline(
                        float(boundary),
                        color="tab:orange",
                        linestyle="--",
                        linewidth=1.2,
                        label="Peak boundaries" if boundary_index == 0 else None,
                    )

        if review_status is None:
            status = "Pass" if has_result else "Failed"
        else:
            status = "Pass" if review_status == "pass" else "Failed"
        axis.set_title(
            f"{peak_label(peak_index)} peak shared outer-wing baseline - {status}"
        )
        axis.set_xlabel("Potential (V)")
        axis.set_ylabel("Current")
        axis.legend(loc="best")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        output_path = _multi_peak_figure_path(
            file_name,
            file_index,
            curve_index,
            peak_index,
            output_directory,
        )
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return output_path
    except Exception as error:
        try:
            plt.close(figure)
        except (NameError, UnboundLocalError):
            pass
        print(
            f"{file_name} curve {int(curve_index) + 1} "
            f"{peak_label(peak_index)} peak figure failed: {error}"
        )
        return None


def process_file_multi(args):
    """Analyze all requested peaks in every curve of one physical file.

    Every baseline algorithm is fitted once from only the two outer wings.
    The same candidate is then screened against every requested peak, and only
    the intersection of those screens contributes to one shared baseline.
    """

    curve_index_offset = 0
    is_curve_task = len(args) == 12
    if len(args) == 10:
        (
            file_name,
            data,
            num_curves,
            file_index,
            cpd_search_model,
            cpd_cost_function,
            peak_region_threshold,
            noise_level,
            fitting_algorithms,
            peak_count,
        ) = args
        figure_output_directory = MULTI_PEAK_FIGURE_ROOT
    elif len(args) == 11:
        (
            file_name,
            data,
            num_curves,
            file_index,
            cpd_search_model,
            cpd_cost_function,
            peak_region_threshold,
            noise_level,
            fitting_algorithms,
            peak_count,
            figure_output_directory,
        ) = args
    elif is_curve_task:
        (
            file_name,
            data,
            num_curves,
            file_index,
            cpd_search_model,
            cpd_cost_function,
            peak_region_threshold,
            noise_level,
            fitting_algorithms,
            peak_count,
            figure_output_directory,
            curve_index_offset,
        ) = args
    else:
        raise ValueError(
            "multi-peak analysis arguments must contain 10, 11, or 12 values"
        )
    if isinstance(peak_count, bool) or not isinstance(
        peak_count, (int, np.integer)
    ) or peak_count < 2:
        raise ValueError("multi-peak processing requires at least two peaks")

    if not is_curve_task:
        _validate_peak_capacity(
            data,
            int(peak_count),
            file_name,
            cpd_search_model,
            cpd_cost_function,
            peak_region_threshold,
            noise_level,
        )
    results_by_peak = [[] for _ in range(int(peak_count))]
    for data_curve_index in range(num_curves):
        curve_index = data_curve_index + int(curve_index_offset)
        try:
            potential = np.asarray(data[0][data_curve_index], dtype=float)
            current = np.asarray(data[1][data_curve_index], dtype=float)
            regions, smoothed = Change_Point_Detection.CPD_multi(
                potential,
                current,
                cpd_search_model,
                cpd_cost_function,
                peak_region_threshold,
                int(peak_count),
                noise_level,
            )
        except Exception as error:
            print(
                f"{file_name} curve {curve_index + 1} multi-peak "
                f"change-point detection failed: {error}"
            )
            potential = np.asarray(
                data[0][data_curve_index], dtype=float
            ).reshape(-1)
            raw_current = np.asarray(
                data[1][data_curve_index], dtype=float
            ).reshape(-1)
            fallback_mask = np.zeros(len(potential), dtype=bool)
            for peak_index in range(int(peak_count)):
                results_by_peak[peak_index].append(
                    _multi_peak_failure(None, potential)
                )
                _save_multi_peak_figure(
                    file_name,
                    file_index,
                    curve_index,
                    peak_index,
                    potential,
                    raw_current,
                    raw_current,
                    None,
                    fallback_mask,
                    output_directory=figure_output_directory,
                )
            continue

        regions = list(regions)
        curve_results = [
            _multi_peak_failure(
                regions[peak_index] if peak_index < len(regions) else None,
                potential,
            )
            for peak_index in range(int(peak_count))
        ]
        mask = np.zeros(len(potential), dtype=bool)
        baselines = []
        baseline_mean = None
        baseline_half_width = None

        try:
            if len(regions) != int(peak_count):
                raise ValueError(
                    "change-point detection did not return every requested peak"
                )
            (
                mask,
                baselines,
                baseline_mean,
                baseline_half_width,
            ) = _fit_shared_multi_peak_baseline(
                potential,
                smoothed,
                regions,
                fitting_algorithms,
            )
            calculated_results = _shared_multi_peak_result_updates(
                potential,
                smoothed,
                regions,
                baseline_mean,
                baseline_half_width,
            )
            for peak_index, updates in enumerate(calculated_results):
                curve_results[peak_index].update(updates)
        except Exception as error:
            print(
                f"{file_name} curve {curve_index + 1} shared baseline "
                f"analysis failed: {error}"
            )

        for peak_index, peak_result in enumerate(curve_results):
            region = regions[peak_index] if peak_index < len(regions) else None
            _save_multi_peak_figure(
                file_name,
                file_index,
                curve_index,
                peak_index,
                potential,
                current,
                smoothed,
                region,
                mask,
                baselines=baselines,
                baseline_mean=baseline_mean,
                baseline_half_width=baseline_half_width,
                output_directory=figure_output_directory,
                review_status=peak_result["review_status"],
            )

        for peak_index, peak_result in enumerate(curve_results):
            results_by_peak[peak_index].append(peak_result)

    return file_name, results_by_peak, file_index


def process_curve_multi(args):
    """Analyze one curve and retain its original file/curve coordinates.

    Keeping this wrapper at module scope makes it picklable by the Windows
    multiprocessing ``spawn`` start method. ``process_file_multi`` remains
    available for legacy callers that intentionally submit an entire file.
    """

    curve_index = int(args[-1])
    file_name, peak_results, file_index = process_file_multi(args)
    if any(len(results) != 1 for results in peak_results):
        raise ValueError("a multi-peak curve task must return one result per peak")
    return (
        file_name,
        curve_index,
        [results[0] for results in peak_results],
        file_index,
    )


def _store_curve_for_peaks(data_save, file_name, curve_key, metadata, peak_count):
    """Duplicate source metadata into each logical per-peak result file."""

    for peak_index in range(peak_count):
        result_name = peak_result_name(file_name, peak_index, peak_count)
        curve = dict(metadata)
        if peak_count > 1:
            curve.update(
                {
                    SOURCE_FILE_KEY: str(file_name),
                    PEAK_NUMBER_KEY: peak_index + 1,
                    PEAK_LABEL_KEY: peak_label(peak_index),
                }
            )
        data_save.setdefault(result_name, {})[curve_key] = curve


def _validate_peak_capacity(
    raw_data,
    peak_count,
    file_name,
    cpd_search_model=None,
    cpd_cost_function=None,
    peak_region_threshold=None,
    noise_level=None,
):
    """Reject sample- or detector-incompatible peak counts before batching."""

    try:
        curves = list(zip(raw_data[0], raw_data[1]))
        lengths = [min(len(potential), len(current)) for potential, current in curves]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"{file_name} does not contain valid curve arrays") from exc
    if not lengths:
        raise ValueError(f"{file_name} does not contain any curves")
    shortest_curve_index = int(np.argmin(lengths))
    shortest_length = lengths[shortest_curve_index]
    max_peak_count = max(0, (shortest_length - 2) // 3)
    if peak_count > max_peak_count:
        raise ValueError(
            f"{file_name} supports at most {max_peak_count} peak(s) because "
            f"its shortest curve contains {shortest_length} samples"
        )

    detector_settings = (
        cpd_search_model,
        cpd_cost_function,
        peak_region_threshold,
        noise_level,
    )
    if peak_count < 2 or any(value is None for value in detector_settings):
        return

    change_point_count = int(peak_count) * 3
    # Capacity depends on the detector's minimum segment size and jump, not on
    # the measured values.  Probe it with a valid synthetic curve of the same
    # length so a corrupt source curve is still handled as a per-curve data
    # failure rather than being mislabeled as an unsupported peak count.
    probe_potential = np.arange(shortest_length, dtype=float)
    probe_phase = np.linspace(0.0, 8.0 * np.pi, shortest_length)
    probe_current = np.sin(probe_phase) + 0.25 * np.cos(0.37 * probe_phase)
    try:
        Change_Point_Detection.CPD_multi(
            probe_potential,
            probe_current,
            cpd_search_model,
            cpd_cost_function,
            peak_region_threshold,
            int(peak_count),
            noise_level,
        )
    except BadSegmentationParameters as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise ValueError(
            f"{file_name} cannot detect {peak_count} peak(s) "
            f"({change_point_count} change points) with the selected "
            f"{cpd_search_model}/{cpd_cost_function} detector: {detail}"
        ) from exc


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


def _normalise_measurement_type(measurement_type):
    """Return the supported batch measurement type."""

    if not isinstance(measurement_type, str):
        raise ValueError("measurement type must be either 'swv' or 'cv'")
    normalised = measurement_type.strip().lower()
    if normalised not in {"swv", "cv"}:
        raise ValueError("measurement type must be either 'swv' or 'cv'")
    return normalised


def _split_cv_curve_collection(source_file, curves):
    """Split every curve in one physical file and group like scan directions."""

    try:
        potentials, currents = curves
    except (TypeError, ValueError) as error:
        raise CVProcessingError(
            f"CV source {source_file} does not contain potential/current arrays."
        ) from error
    if len(potentials) != len(currents) or not potentials:
        raise CVProcessingError(
            f"CV source {source_file} must contain paired potential/current curves."
        )

    branch_pairs = []
    for curve_number, (potential, current) in enumerate(
        zip(potentials, currents), start=1
    ):
        try:
            sample_count = len(potential)
            signal = CVSignal(
                source_path=str(source_file),
                sequence=np.arange(sample_count, dtype=float),
                potential=np.asarray(potential, dtype=float),
                current=np.asarray(current, dtype=float),
            )
            branch_pairs.append(split_cv_signal(signal))
        except (CVProcessingError, TypeError, ValueError) as error:
            raise CVProcessingError(
                f"CV source {source_file}, curve {curve_number}: {error}"
            ) from error

    groups = []
    for branch_index, direction in enumerate(("oxidation", "reduction")):
        branches = [pair[branch_index] for pair in branch_pairs]
        logical_name = derived_branch_filename(source_file, direction)
        grouped_curves = [
            [branch.potential.tolist() for branch in branches],
            [branch.pipeline_current.tolist() for branch in branches],
        ]
        groups.append((logical_name, grouped_curves, branches))
    return groups


def _read_standard_cv_groups(source_file):
    """Read the public three-column CV format as one logical curve."""

    oxidation, reduction = load_cv_branches(source_file)
    groups = []
    for branch in (oxidation, reduction):
        groups.append(
            (
                branch.derived_filename,
                branch.as_pipeline_curves(),
                [branch],
            )
        )
    return groups


def _cv_curve_metadata(branch, base_metadata=None):
    metadata = dict(base_metadata or {})
    metadata.update(branch.metadata)
    metadata.update(
        {
            "Raw Poetntial ": branch.potential.tolist(),
            "Raw Current": branch.pipeline_current.tolist(),
            CV_ORIGINAL_CURRENT_KEY: branch.original_current.tolist(),
        }
    )
    return metadata


def _apply_cv_peak_sign(curve):
    """Store the signed peak current while retaining positive peak magnitude."""

    try:
        multiplier = int(curve.get(CV_CURRENT_SIGN_KEY, 1))
        peak_value = float(curve.get("Peak Value ", 0))
    except (TypeError, ValueError):
        multiplier = 1
        peak_value = 0.0
    if multiplier not in {-1, 1}:
        multiplier = 1
    curve[SIGNED_PEAK_CURRENT_KEY] = peak_value * multiplier
    return curve


def data_analysis(
    data_fromGUI,
    CPD_SM,
    CPD_CF,
    Alg_peak,
    Alg_noise_level,
    Alg_fitting_set,
    progress_callback=None,
    peak_count=1,
    measurement_type="swv",
):
        # print(result)
        freeze_support()
        if isinstance(peak_count, bool) or not isinstance(
            peak_count, (int, np.integer)
        ) or peak_count < 1:
            raise ValueError("peak count must be a positive integer")
        peak_count = int(peak_count)
        measurement_type = _normalise_measurement_type(measurement_type)
        if measurement_type == "cv" and peak_count != 1:
            raise ValueError("CV analysis requires exactly one peak per scan branch")
        #data_fromGUI = {'pssession': [{'file_type': 'pssession', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/3.pssession', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/2.pssession'], 'frequency': None, 'amplitude': None}], 'csv': [{'file_type': 'csv', 'file_names': ['C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_30mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/20Hz_50mV.csv', 'C:/Users/jyk98/Desktop/Basepeak/PSPythonSDK/CH03/40Hz_10mV.csv'], 'frequency': 0.0, 'amplitude': 0.0}]}

        if not os.path.exists('Fig_Saved'):
            os.makedirs('Fig_Saved')

        multi_peak_figure_directory = None
        if peak_count > 1:
            # Multi-peak analysis always evaluates the complete approved
            # 30-algorithm library. Keep each run's shared-baseline peak views
            # together so later runs cannot overwrite or mix with it.
            Alg_fitting_set = list(MULTI_PEAK_BASELINE_ALGORITHMS)
            run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            multi_peak_figure_directory = os.path.abspath(
                os.path.join(MULTI_PEAK_FIGURE_ROOT, run_stamp)
            )
            os.makedirs(multi_peak_figure_directory, exist_ok=True)

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
                    file_read_result = (
                        read_pssession_file(file, trim_edges=False)
                        if measurement_type == "cv"
                        else read_pssession_file(file)
                    )
                except Exception as error:
                    if measurement_type == "cv":
                        raise ValueError(
                            f"Could not read CV pssession file {file}: {error}"
                        ) from error
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
                if measurement_type == "cv":
                    try:
                        cv_groups = _split_cv_curve_collection(
                            file, file_read_result[0]
                        )
                    except CVProcessingError as error:
                        raise ValueError(f"Could not split CV file {file}: {error}") from error
                    for logical_name, grouped_curves, branches in cv_groups:
                        _validate_peak_capacity(
                            grouped_curves,
                            1,
                            logical_name,
                            CPD_SM,
                            CPD_CF,
                            Alg_peak,
                            Alg_noise_level,
                        )
                        file_name.append(logical_name)
                        raw_data.append(grouped_curves)
                        num_curves.append(len(branches))
                        for i, branch in enumerate(branches):
                            _store_curve_for_peaks(
                                data_save,
                                logical_name,
                                'Curve No. '+str(i+1),
                                _cv_curve_metadata(
                                    branch,
                                    {
                                        'Date and time measurement': file_read_result[1][i],
                                        'Channel ': file_read_result[6][i],
                                        'Frequence ': file_read_result[3][i],
                                        'Amplitude ': file_read_result[4][i],
                                        'E Step ': file_read_result[5][i],
                                    },
                                ),
                                1,
                            )
                    continue
                _validate_peak_capacity(
                    file_read_result[0],
                    peak_count,
                    file,
                    CPD_SM,
                    CPD_CF,
                    Alg_peak,
                    Alg_noise_level,
                )
                file_name.append(file)
                raw_data.append(file_read_result[0])
                a = file_read_result[2]#numger of curves
                # print(file,a,'curves')
                num_curves.append(a )

                for i in range(a):
                    _store_curve_for_peaks(
                        data_save,
                        file,
                        'Curve No. '+str(i+1),
                        {
                            'Date and time measurement': file_read_result[1][i],
                            'Channel ': file_read_result[6][i],
                            'Frequence ': file_read_result[3][i],
                            'Amplitude ': file_read_result[4][i],
                            'E Step ': file_read_result[5][i],
                            'Raw Poetntial ': file_read_result[0][0][i],
                            'Raw Current': file_read_result[0][1][i],
                        },
                        peak_count,
                    )

        if files[1] and len(files[1][0]) > 0:
            print('Reading CSV File')
            for file in files[1][0]:
                if measurement_type == "cv":
                    try:
                        try:
                            cv_groups = _read_standard_cv_groups(file)
                            cv_times = [""]
                        except CVProcessingError as standard_error:
                            try:
                                file_read_result = read_csv_file(file, trim_edges=False)
                                cv_groups = _split_cv_curve_collection(
                                    file, file_read_result[0]
                                )
                                cv_times = [
                                    value.strftime('%Y-%m-%d %H:%M:%S')
                                    for value in file_read_result[1]
                                ]
                            except Exception as legacy_error:
                                raise ValueError(
                                    f"standard CV parser: {standard_error}; "
                                    f"A-PACE CSV parser: {legacy_error}"
                                ) from legacy_error
                    except Exception as error:
                        raise ValueError(
                            f"Could not read or split CV file {file}: {error}"
                        ) from error
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
                    for logical_name, grouped_curves, branches in cv_groups:
                        _validate_peak_capacity(
                            grouped_curves,
                            1,
                            logical_name,
                            CPD_SM,
                            CPD_CF,
                            Alg_peak,
                            Alg_noise_level,
                        )
                        file_name.append(logical_name)
                        raw_data.append(grouped_curves)
                        num_curves.append(len(branches))
                        for i, branch in enumerate(branches):
                            measured_at = cv_times[i] if i < len(cv_times) else ""
                            _store_curve_for_peaks(
                                data_save,
                                logical_name,
                                'Curve No. '+str(i+1),
                                _cv_curve_metadata(
                                    branch,
                                    {'Date and time measurement': measured_at},
                                ),
                                1,
                            )
                    continue
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
                _validate_peak_capacity(
                    file_read_result[0],
                    peak_count,
                    file,
                    CPD_SM,
                    CPD_CF,
                    Alg_peak,
                    Alg_noise_level,
                )
                file_name.append(file)
                raw_data.append(file_read_result[0])
                a = file_read_result[2]
                print(file,a,'curves')
                Time_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in  file_read_result[1]]
                num_curves.append(a )
                for i in range(a):
                    _store_curve_for_peaks(
                        data_save,
                        file,
                        'Curve No. '+str(i+1),
                        {
                            'Date and time measurement': Time_str[i],
                            'Raw Poetntial ': file_read_result[0][0][i],
                            'Raw Current': file_read_result[0][1][i],
                        },
                        peak_count,
                    )

        if peak_count == 1:
            args = [
                (
                    file_name[i], raw_data[i], num_curves[i], i, CPD_SM,
                    CPD_CF, Alg_peak, Alg_noise_level, Alg_fitting_set,
                    measurement_type,
                    data_save[file_name[i]]["Curve No. 1"].get(CV_CURRENT_SIGN_KEY, 1),
                )
                for i in range(len(raw_data))
            ]
            worker = process_file
            work_unit = "valid file(s)"
        else:
            args = [
                (
                    file_name[i],
                    [
                        [raw_data[i][0][current_curve_index]],
                        [raw_data[i][1][current_curve_index]],
                    ],
                    1,
                    i,
                    CPD_SM,
                    CPD_CF, Alg_peak, Alg_noise_level, Alg_fitting_set,
                    peak_count, multi_peak_figure_directory,
                    current_curve_index,
                )
                for i in range(len(raw_data))
                for current_curve_index in range(num_curves[i])
            ]
            worker = process_curve_multi
            work_unit = "curve(s)"

        if not args:
            raise ValueError("No valid input files were provided for analysis")
        _report_analysis_progress(
            progress_callback,
            25,
            f"Analyzing 0 of {len(args)} {work_unit}...",
            0,
            len(args),
        )
        available_cpus = cpu_count() or 1
        num_cpus = max(1, min(len(args), available_cpus - 1))

        print(f"Using {num_cpus} CPUs for Analysis" )

        with Pool(processes=num_cpus) as pool:
            #results = pool.map(process_file, args)

            results = []
            # start_time = time.time()
            for result in tqdm(pool.imap_unordered(worker, args), total=len(args)):
                results.append(result)
                completed_files = len(results)
                _report_analysis_progress(
                    progress_callback,
                    25 + 65 * completed_files / len(args),
                    f"Analyzed {completed_files} of {len(args)} {work_unit}.",
                    completed_files,
                    len(args),
                )
                # elapsed_time = time.time() - start_time
                # tqdm.write(f"Elapsed time: {elapsed_time:.2f} seconds")


        if peak_count == 1:
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
                    if measurement_type == "cv":
                        _apply_cv_peak_sign(
                            data_save[file_name_save][
                                'Curve No. '+str(curve_index+1)
                            ]
                        )
        else:
            for (
                file_name_save,
                curve_index,
                curve_results,
                file_index,
            ) in results:
                for peak_index, peak_result in enumerate(curve_results):
                    result_name = peak_result_name(
                        file_name_save, peak_index, peak_count
                    )
                    curve = data_save[result_name][
                        'Curve No. '+str(curve_index+1)
                    ]
                    curve.update(peak_result)
                    curve['Concentration'] = "undefined"


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
