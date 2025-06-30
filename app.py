''' APACE GUI '''

import sys
from flask import Flask, Response, render_template, request, redirect, send_file, url_for, jsonify
import json
import os
import subprocess
import logging
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import demo
import ast
import atexit
import threading
from werkzeug.serving import make_server
from datetime import datetime  # Added for timestamp
import numpy as np
from real_time_analysis import run_real_time_analysis
from CPD_change import process_file
app = Flask(__name__)
PALETTE = px.colors.qualitative.Pastel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
deleted_graphs: list[tuple[str, str, dict]] = []

# File paths
DATA_DIR = 'database'
DEMO_JSON_PATH = os.path.join(DATA_DIR, 'results.json')
DATA_TABLE_PATH = os.path.join(DATA_DIR, 'data_table.json')
UPLOADED_FILES_PATH = os.path.join(DATA_DIR, 'uploaded_files.json')
UPLOADED_FOLDER_PATH = os.path.join(DATA_DIR, 'uploaded_folder.json')
PASS_GRAPHS_PATH = os.path.join(DATA_DIR, 'pass_graphs.json')
FAIL_GRAPHS_PATH = os.path.join(DATA_DIR, 'fail_graphs.json')
ALGORITHM_SETTINGS_PATH = 'Algorithm Setting.json'
REAL_TIME_FOLDER_PATH = os.path.join(DATA_DIR, 'real_time_folder_path.json')
ALLOWED_EXTENSIONS = {'csv', 'pssession'}

# Utility Functions
def read_json(file_path):
    """Read JSON from file, return empty dict on error or if file doesn't exist."""
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return {}

def write_json(file_path, data):
    """Write JSON to file, create directories if needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing to {file_path}: {e}")
        raise

def allowed_file(filename, file_type):
    """Check if filename has allowed extension and matches file_type."""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and 
            filename.rsplit('.', 1)[1].lower() == file_type)

def initialize_data_table(json_dataset):
    """Initialize data_table.json with data from json_dataset."""
    data_table = {}
    for file, curves in json_dataset.items():
        data_table[file] = {}
        for curve, data in curves.items():
            data_table[file][curve] = {
                "Date and time measurement": data.get("Date and time measurement", ""),
                "Frequence ": data.get("Frequence ", ""),
                "Amplitude ": data.get("Amplitude ", ""),
                "Peak Value ": data.get("Peak Value ", ""),
                "Channel ": data.get("Channel ", ""),
                "Concentration": data.get("Concentration", "")
            }
    write_json(DATA_TABLE_PATH, data_table)
    return data_table

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_folder_path', methods=['GET', 'POST'])
def save_folder_path():
    data = request.get_json()
    print('data', data)
    folder_path = data.get('folder_path')
    
    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    # Save the folder path to a JSON file
    with open('database/real_time_folder_path.json', 'w') as f:
        json.dump({"folder_path": folder_path}, f)

    return jsonify({"status": "Folder path saved"}), 200

def start_real_time_analysis(SR_weight, noise_level, Threshold, SlipWindow):
    # 1) read folder path
    path = json.load(open(REAL_TIME_FOLDER_PATH))["folder_path"]
    if not path:
        logging.error("Real-time folder path missing")
        return

    # 2) load your alg settings
    data_algs = json.load(open(ALGORITHM_SETTINGS_PATH, 'r'))
    name      = str(int(SR_weight/0.01))
    CPD_SM    = data_algs[name]["CPD Search Model"]
    CPD_CF    = data_algs[name]["CPD Cost Function"]
    peak_ratio= Threshold                # same as your 5th param
    fitlist   = ast.literal_eval(data_algs[name]["Baseline Fitting Algorithms"])
    fitlist_json = json.dumps(fitlist)

    # 3) spawn the exact same 7 args in the same order
    script = os.path.join(os.path.dirname(__file__), "real_time_analysis.py")
    subprocess.Popen([
        sys.executable,        # ensures the same venv python is used
        script,                # your real_time_analysis.py file
        path,                  # 1) folder path
        str(SlipWindow),       # 2) slip window
        CPD_SM,                # 3) CPD search model
        CPD_CF,                # 4) CPD cost function
        str(peak_ratio),       # 5) peak ratio
        str(noise_level),      # 6) noise level
        fitlist_json           # 7) fitting algorithm list
    ], cwd=os.path.dirname(__file__))

@app.route('/real_time', methods=['POST', 'GET'])
def real_time():
    if request.method == 'POST':
        try:
            data = request.get_json()
            print('param data')
            print(data)
            if not data:
                return jsonify({"error": "No data received"}), 400

            SR_weight = float(data.get('successWeight', 0.5))
            noise_level = int(data.get('noiseLevel', 2))
            Threshold = float(data.get('Threshold', 0.65))
            SlipWindow = int(data.get('SlipWindow', 5))

            # Just call start_real_time_analysis, which will now launch subprocess
            start_real_time_analysis(SR_weight, noise_level, Threshold, SlipWindow)

            return jsonify({"status": "success", "message": "Real-time analysis started"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # If GET request, just show the page
    files = read_json(UPLOADED_FOLDER_PATH)
    try:
        with open('uploaded_folder.json', 'r') as f:
            uploaded_files = json.load(f)
            files.update(uploaded_files)
    except FileNotFoundError:
        pass

    return render_template('real_time.html', files=files)

@app.route('/launch_folder_gui', methods=['GET', 'POST'])
def launch_folder_gui():
    logging.debug("Launching GUI...")
    subprocess.Popen([sys.executable, 'folder_upload.py'])
    return jsonify({"status": "GUI launched"})

@app.route('/post_exp', methods=['GET', 'POST'])
def post_exp():
    return render_template('post_exp.html')

@app.route('/launch_file_gui', methods=['POST'])
def launch_file_gui():
    logging.debug("Launching GUI...")
    subprocess.Popen([sys.executable, 'file_upload.py'])
    return jsonify({"status": "GUI launched"})

@app.route('/post_exp/upload', methods=['GET', 'POST'])
def upload():
    uploaded_files = read_json(UPLOADED_FILES_PATH) or {"csv": []}
    return render_template('upload.html', files=uploaded_files)

def is_json_files_empty(file_path):
    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                # Check if the JSON data is empty
                if not data:  # This checks for {}, [], or None
                    return True
                return False
            except json.JSONDecodeError:
                # Handle case where file contents are invalid JSON
                return True
    else:
        # File does not exist or is empty
        return True
    
@app.route('/post_exp/upload/data', methods=['POST'])
def analyze():
    # Get JSON payload from the request
    print(request)
    data = request.get_json()
    print(data)

    # Extract parameters from the payload
    SR_weight = float(data.get('successWeight', 0.5))  # Default to 0.5 if not provided
    noise_level = int(data.get('noiseLevel', 2))  # Default to 3 if not provided
    Threshold = float(data.get('Threshold', 0.65))
    print(SR_weight, noise_level, Threshold)  # Debugging statement

    # Prepare the dictionary to save
    parameters = {
        "successWeight": SR_weight,
        "noiseLevel": noise_level,
        "Threshold": Threshold
    }
    print("Parameters to save:", parameters)  # Debugging statement

    # Save parameters to parameters.json
    with open('database/parameters.json', 'w', encoding='utf-8') as file:
        json.dump(parameters, file, indent=4)

    uploaded = read_json(UPLOADED_FILES_PATH)

    data_fromGUI = {
        'csv':      {'file_names': uploaded.get('csv', [])},
        'pssession':{'file_names': uploaded.get('pssession', [])}
    }

    with open('Algorithm Setting.json', 'r', encoding='utf-8') as file:
        data_algs = json.load(file)
    name = str(int(SR_weight / 0.01))
    
    # Run the data analysis function
    # demo_noPssession.data_analysis(data_fromGUI)
    
    
    start_time = datetime.now()
    demo.data_analysis(data_fromGUI,
                data_algs[name][ "CPD Search Model"],
                data_algs[name][ "CPD Cost Function"],
                Threshold,
                noise_level,
                list(ast.literal_eval(data_algs[name][ "Baseline Fitting Algorithms"]) ))
    print('Time spent: ',datetime.now()- start_time )
    pass_data = {}
    fail_data = {}
    demo_data = read_json(DEMO_JSON_PATH)
    for fname, curves in demo_data.items():
        for cno, cd in curves.items():
            if cd.get('Peak Value ', 0) == 0:
                fail_data.setdefault(fname, {})[cno] = cd
            else:
                pass_data.setdefault(fname, {})[cno] = cd

    write_json(PASS_GRAPHS_PATH, pass_data)
    write_json(FAIL_GRAPHS_PATH, fail_data)

    # Return a JSON response indicating success
    return jsonify({"status": "success", "message": "Data analysis complete"}), 200

@app.route('/post_exp/delete-multiple', methods=['POST'])
def delete_multiple_files():
    try:
        files_to_delete = request.form.getlist('delete_files')
        uploaded_files = read_json(UPLOADED_FILES_PATH)
        for file_type in uploaded_files:
            uploaded_files[file_type] = [f for f in uploaded_files[file_type] if f not in files_to_delete]
        write_json(UPLOADED_FILES_PATH, uploaded_files)
        return redirect(url_for('upload'))
    except Exception as e:
        logging.error(f"Error deleting files: {e}")
        return jsonify({"error": "Failed to delete files"}), 500

@app.route('/post_exp/delete-all', methods=['POST'])
def delete_all_files():
    try:
        write_json(UPLOADED_FILES_PATH, {"csv": [], "pssession": []})
        return redirect(url_for('upload'))
    except Exception as e:
        logging.error(f"Error deleting all files: {e}")
        return jsonify({"error": "Failed to delete files"}), 500

def draw_graph(file_name, curve_no, curve_data):
    """Generate a Plotly figure for a curve."""
    try:
        fig = go.Figure()
        raw_potential = curve_data.get('Raw Poetntial ')
        raw_current = curve_data.get('Raw Current')
        cp_indexes = curve_data.get('Change Point Indexes ')
        cp_values = curve_data.get('Change Point Values ')

        if isinstance(raw_potential, list) and isinstance(raw_current, list):
            fig.add_trace(go.Scatter(x=raw_potential, y=raw_current, mode='lines', name='Raw Data', line=dict(color='red')))
        else:
            logging.warning(f"Invalid data format for {curve_no} in {file_name}")
            return None, [], [], [], []

        if 'Baseline Mean ' in curve_data:
            baseline_mean = curve_data['Baseline Mean ']
            if isinstance(baseline_mean, list):
                fig.add_trace(go.Scatter(x=raw_potential, y=baseline_mean, mode='lines', name='Baseline', line=dict(color='blue')))
                ci_lower = curve_data.get('95\\% Confidence Interval of Baseline: ', [])
                ci_upper = ci_lower
                if isinstance(ci_lower, list) and isinstance(ci_upper, list):
                    fig.add_trace(go.Scatter(x=raw_potential, y=ci_lower, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                    fig.add_trace(go.Scatter(x=raw_potential, y=ci_upper, fill='tonexty', mode='none', name='95% CI (Baseline)', fillcolor='rgba(0,0,255,0.1)', line=dict(color='rgba(0,0,0,0)')))
        
        if 'Change Point Indexes ' in curve_data and isinstance(cp_indexes, list) and isinstance(cp_values, list):
            for cp in cp_values:
                fig.add_vline(x=cp, line=dict(color='red', width=2), annotation_text=str(cp), annotation_position='top right', name=f'Change Point at {cp}')

        peak_array = []
        ci_lower_peak = []
        ci_upper_peak = []
        if isinstance(raw_current, list) and isinstance(baseline_mean, list):
            for i in range(len(raw_potential)):
                peak_array.append(raw_current[i] - baseline_mean[i])
                ci_lower_peak.append(raw_current[i] - curve_data.get('95\\% Confidence Interval of Baseline: ', [0])[i])
                ci_upper_peak.append(raw_current[i] - curve_data.get('95\\% Confidence Interval of Baseline: ', [0])[i])
            fig.add_trace(go.Scatter(x=raw_potential, y=peak_array, mode='lines', name='Peak Curve', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=raw_potential, y=ci_lower_peak, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=raw_potential, y=ci_upper_peak, fill='tonexty', mode='none', name='95% CI (Peak)', fillcolor='rgba(0,255,0,0.2)', line=dict(color='rgba(0,0,0,0)')))

        if 'Peak Value ' in curve_data:
            peak_value = curve_data['Peak Value ']
            peak_location = curve_data['Peak Location: ']
            if not isinstance(peak_value, list):
                peak_value = [peak_value]
            if not isinstance(peak_location, list):
                peak_location = [peak_location]
            if isinstance(peak_value, list) and isinstance(peak_location, list):
                fig.add_trace(go.Scatter(x=peak_location, y=peak_value, mode='markers', name='Peak', marker=dict(color='green', size=10)))
                for loc, val in zip(peak_location, peak_value):
                    fig.add_annotation(x=loc, y=val, text=f'{val:.2f}', showarrow=True, arrowhead=2, ax=0, ay=-20)

        fig.update_layout(title=f"{file_name} - {curve_no}", xaxis_title='Potential(V)', yaxis_title='Current(uA)', margin=dict(l=20, r=20, t=30, b=20))
        return fig.to_html(full_html=False), raw_potential, peak_array, ci_lower_peak, ci_upper_peak
    except Exception as e:
        logging.error(f"Error processing curve {curve_no} in {file_name}: {e}")
        return None, [], [], [], []

@app.route('/post_exp/pass', methods=['GET', 'POST'])
def pass_graphs():
    try:
        demo_data = read_json(DEMO_JSON_PATH)
        graphs = []
        page = int(request.args.get('page', 1))
        per_page = 15

        for file_name, data in demo_data.items():
            for curve_no, curve_data in data.items():
                result = draw_graph(file_name, curve_no, curve_data)
                if result:
                    graph_html, raw_potential, peak_array, ci_lower_peak, ci_upper_peak = result
                    peak_height = max(peak_array) if peak_array else 0
                    graphs.append({
                        'html': graph_html,
                        'raw_potential': raw_potential,
                        'ci_lower_peak': ci_lower_peak,
                        'ci_upper_peak': ci_upper_peak,
                        'file_name': file_name,
                        'curve_no': curve_no,
                        'peak_height': peak_height,
                        'frequency': curve_data.get('Frequence ', 0),
                        'concentration': curve_data.get('Concentration ', None)
                    })

        if request.method == 'POST':
            filters = request.json
            filtered_graphs = [
                g for g in graphs
                if (not filters.get('peakHeight') or (filters['peakHeight']['min'] <= g['peak_height'] <= filters['peakHeight']['max'])) and
                   (not filters.get('frequency') or (filters['frequency']['min'] <= g['frequency'] <= filters['frequency']['max'])) and
                   (not filters.get('concentration') or (g['concentration'] is not None and filters['concentration']['min'] <= g['concentration'] <= filters['concentration']['max']))
            ]
            filtered_graphs = list({g['file_name'] + g['curve_no']: g for g in filtered_graphs}.values())
            total_graphs = len(filtered_graphs)
            total_pages = (total_graphs + per_page - 1) // per_page
            start_index = (filters.get('page', 1) - 1) * per_page
            end_index = start_index + per_page
            paginated_graphs = filtered_graphs[start_index:end_index]
            return jsonify({
                "graphs": paginated_graphs,
                "current_start": start_index + 1,
                "current_end": min(end_index, total_graphs),
                "total_graphs": total_graphs,
                "page": filters.get('page', 1),
                "total_pages": total_pages
            })

        concentration_missing = any(graph['concentration'] is None for graph in graphs)
        total_graphs = len(graphs)
        total_pages = (total_graphs + per_page - 1) // per_page
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        current_graphs = graphs[start_index:end_index]

        return render_template(
            'pass.html',
            graphs=current_graphs,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            total_graphs=total_graphs,
            current_start=start_index + 1,
            current_end=min(end_index, total_graphs),
            concentration_missing=concentration_missing
        )
    except Exception as e:
        logging.error(f"Error in pass_graphs: {e}")
        return jsonify({"error": "Failed to process request"}), 500

def draw_fail_graph(file_name, curve_no, curve_data):
    peak_value = curve_data.get('Peak Value ', 0)  # Default to 0 if not found

    if peak_value != 0:
        # Call draw_graph for a complete graph if Peak Value is not zero
        return draw_graph(file_name, curve_no, curve_data)[0]
    
    fig = go.Figure()
    try:
        raw_potential = curve_data.get('Raw Poetntial ')
        raw_current = curve_data.get('Raw Current')

        # Plot raw data
        if isinstance(raw_potential, list) and isinstance(raw_current, list):
            fig.add_trace(go.Scatter(x=raw_potential, y=raw_current, mode='lines', name='Raw Data', line=dict(color=PALETTE[0])))
        else:
            print(f"Invalid data format for curve {curve_no} in file {file_name}.")
            return None

        fig.update_layout(
            title=f"{file_name} - {curve_no} (Failed Graph)",
            xaxis_title='Potential(V)',
            yaxis_title='Current(µA)',
            margin=dict(l=20, r=20, t=30, b=20)
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"Error drawing failed graph for {curve_no} in {file_name}: {e}")
        return None
    
@app.route('/post_exp/fail')
def fail_graphs():
    fail_data = read_json(FAIL_GRAPHS_PATH)
    graphs = []

    for file_name, curves in fail_data.items():
        for curve_no, curve_data in curves.items():
            graph_html = draw_fail_graph(file_name, curve_no, curve_data)
            if graph_html:
                graphs.append({
                    'html': graph_html,
                    'file_name': file_name,
                    'curve_no': curve_no
                })

    return render_template('fail.html', graphs=graphs)

@app.route('/post_exp/delete_graphs', methods=['POST'])
def delete_graphs():
    # global deleted_graphs
    selected_graphs = request.json['graphs']
    demo_data = read_json(DEMO_JSON_PATH)
    fail_data = read_json(FAIL_GRAPHS_PATH)
    
    for graph_id in selected_graphs:
        if '_Curve No. ' in graph_id:
            file_name, curve_no_suffix = graph_id.rsplit('_Curve No. ', 1)
            file_name = file_name.strip()
            curve_no  = 'Curve No. ' + curve_no_suffix.strip()
        else:
            continue
        if file_name in demo_data and curve_no in demo_data[file_name]:
            curve_data = demo_data[file_name][curve_no]

            # Add to fail_graphs.json
            if file_name not in fail_data:
                fail_data[file_name] = {}
            fail_data[file_name][curve_no] = curve_data

            # Remove from results.json
            del demo_data[file_name][curve_no]
            if not demo_data[file_name]:
                del demo_data[file_name]

    write_json(DEMO_JSON_PATH, demo_data)
    write_json(FAIL_GRAPHS_PATH, fail_data)

    return jsonify(success=True)

@app.route('/post_exp/restore_graphs', methods=['POST'])
def restore_graphs(): #check falled 
    global deleted_graphs
    selected_graphs = request.json['graphs']  # List of selected graphs
    restored_graphs_data = []  # To track successfully restored graphs

    demo_data = read_json(DEMO_JSON_PATH)
    fail_data = read_json(FAIL_GRAPHS_PATH)
    pass_data = read_json(PASS_GRAPHS_PATH)

    for graph_id in selected_graphs:
        file_name, curve_no = graph_id.split('_', 1)
        if file_name in fail_data and curve_no in fail_data[file_name]:
            curve_data = fail_data[file_name][curve_no]

            # Check and reprocess the graph
            peak_value = curve_data.get('Peak Value ', 0)
            if peak_value == 0:
                # Use a different color to signal the rerun
                graph_html = draw_fail_graph(file_name, curve_no, curve_data)
            else:
                # Use the regular draw_graph function
                graph_html, raw_potential, peak_array, ci_lower_peak, ci_upper_peak = draw_graph(file_name, curve_no, curve_data)

                # Move to pass_data if peak_value is valid
                if file_name not in pass_data:
                    pass_data[file_name] = {}
                pass_data[file_name][curve_no] = curve_data

                # Remove from fail_data
                del fail_data[file_name][curve_no]
                if not fail_data[file_name]:  # Remove empty entries
                    del fail_data[file_name]

                restored_graphs_data.append({
                    'file_name': file_name,
                    'curve_no': curve_no,
                    'html': graph_html,
                })

    # Save updated JSON files
    write_json(PASS_GRAPHS_PATH, pass_data)
    write_json(FAIL_GRAPHS_PATH, fail_data)

    return jsonify(success=True, restored_graphs=restored_graphs_data)

# Route for overlaying selected graphs
@app.route('/post_exp/overlay_graphs', methods=['POST'])
def overlay_graphs():
    print('request', request.json)
    selected_graphs = request.json['graphs']
    print('selected graph', selected_graphs)

    demo_data = read_json(DEMO_JSON_PATH)
    # print('demo_data', demo_data)
    # print('json_dataset', json_dataset)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for graph_id in selected_graphs:
        if graph_id != 'on':
            try:
                # Split the graph_id to extract file_name and curve_no
                if '_Curve No.' in graph_id:
                    file_name, curve_no = graph_id.rsplit('_Curve No. ', 1)
                    curve_no = 'Curve No. ' + curve_no.strip()
                    file_name = file_name.strip()
                else:
                    print(f"Unexpected format for graph_id: {graph_id}")
                    continue

                # Fetch curve_data safely
                print('file_name', file_name, 'curve_no', curve_no)
                if file_name in demo_data and curve_no in demo_data[file_name]:
                    curve_data = demo_data[file_name][curve_no]
                else:
                    print(f"KeyError: Missing data for file {file_name} and curve {curve_no}")
                    continue

                # Process curve_data
                _, raw_potential, peak_array, ci_lower_peak, ci_upper_peak = draw_graph(file_name, curve_no, curve_data)
                fig.add_trace(go.Scatter(x=raw_potential, y=peak_array, mode='lines', 
                                        name=f'{file_name} {curve_no}', line=dict(width=1.5)), secondary_y=False)
            except KeyError as e:
                print(f"KeyError: {e} for file {file_name} and curve {curve_no}")

    fig.update_layout(
        title="Overlayed Peak Curves",
        xaxis_title='Potential(V)',
        yaxis_title='Current(uA)',
        margin=dict(l=50, r=50, t=50, b=50),  # Increase margins for better spacing
        legend=dict(
            x=0,
            y=-0.2,
            orientation="h"  # Place legend horizontally below the graph
        )
    )
    graph_json = fig.to_json()
    return jsonify(graph_json)

# Route for updating graphs with user-defined ranges
@app.route('/post_exp/update_graphs', methods=['POST', 'GET'])
def update_graphs():
    selected_graphs = request.json['graphs']
    print("Selected graphs for update:", selected_graphs)

    file_name = ""
    curve_no = 0
    for graph_id in selected_graphs:
        if graph_id != 'on':
            try:
                file_name, curve_no = graph_id.split('_Curve No.')
                curve_no = int(curve_no.strip())-1
            except ValueError:
                print(f"Invalid graph_id format: {graph_id}")
                return jsonify({"error": f"Invalid graph_id format: {graph_id}"}), 400
    print('file_name', file_name, 'curve_no', curve_no)

    left_val = float(request.json['left_val'])
    # if not left_val:
    #     return jsonify({"error": "left value is required"}), 400
    right_val = float(request.json['right_val'])
    # if not right_val:
    #     return jsonify({"error": "right value is required"}), 400
    # print('left', left_val, 'right', right_val)

    params = read_json('database/parameters.json')
    # Prepare data structure for analysis
    SR_weight = float(params.get('successWeight', 0.5))  # Default to 0.5 if not provided
    noise_level = int(params.get('noiseLevel', 2))  # Default to 3 if not provided
    print(f"SR_weight: {SR_weight}, noise_level: {noise_level}")

    with open('Algorithm Setting.json', 'r', encoding='utf-8') as file:
        data_algs = json.load(file)
    
    name = str(int(SR_weight / 0.01))
    print(f"Algorithm name: {name}")

    # Run the data analysis function
    args = (file_name, # file name
            curve_no, # curve index
            [left_val, right_val],  # change point value
            list(ast.literal_eval(data_algs[name][ "Baseline Fitting Algorithms"]) ), # fitting algorithms
            noise_level) # noise level
    print(args) 
    process_file(args)
    # return jsonify({"message": "Range values updated"}), 200

    demo_data = read_json(DEMO_JSON_PATH)

    pass_data, fail_data = {}, {}
    for fname, curves in demo_data.items():
        for cno, cd in curves.items():
            if cd.get('Peak Value ', 0):
                pass_data.setdefault(fname, {})[cno] = cd
            else:
                fail_data.setdefault(fname, {})[cno] = cd
    write_json(PASS_GRAPHS_PATH, pass_data)
    write_json(FAIL_GRAPHS_PATH, fail_data)

    return render_template('fail.html', demo_data=demo_data)


lock = threading.Lock()

@app.route('/post_exp/data-table', methods=['GET', 'POST'])
def data_table():
    try:
        if request.method == 'POST':
            updates = request.json
            if not updates:
                return jsonify({"error": "Invalid JSON format"}), 400

            with lock:
                data_table = read_json(DATA_TABLE_PATH)
                for file, curves in updates.items():
                    if file not in data_table:
                        data_table[file] = {}
                    for curve, values in curves.items():
                        if curve not in data_table[file]:
                            data_table[file][curve] = {
                                "Date and time measurement": "",
                                "Frequence ": "",
                                "Amplitude ": "",
                                "Peak Value ": "",
                                "Channel ": "",
                                "Concentration": ""
                            }
                        # ⭐ Update values
                        data_table[file][curve].update({
                            "Date and time measurement": values.get("date", data_table[file][curve].get("Date and time measurement", "")),
                            "Frequence ": values.get("frequency", ""),
                            "Amplitude ": values.get("amplitude", ""),
                            "Concentration": values.get("concentration", "")
                        })

                write_json(DATA_TABLE_PATH, data_table)

            return jsonify({"message": "Data saved successfully"}), 200

        # GET method: render page
        json_dataset = read_json(DEMO_JSON_PATH)
        data_table = read_json(DATA_TABLE_PATH)
        if not data_table or not any(data_table.values()):
            data_table = initialize_data_table(json_dataset)
        return render_template('data_table.html', demo_data=data_table)

    except Exception as e:
        logging.error(f"Error in data_table: {e}")
        return render_template('error.html', error_message="Failed to load data table"), 500


@app.route('/post_exp/export-table', methods=['GET'])
def export_table():
    try:
        data = read_json(DATA_TABLE_PATH)
        rows = [
            {
                "File Name": file,
                "Curve No.": curve,
                "Frequency": values.get("Frequence ", ""),
                "Amplitude": values.get("Amplitude ", ""),
                "Concentration": values.get("Concentration", "")
            }
            for file, curves in data.items()
            for curve, values in curves.items()
        ]
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=data_table_export.csv"}
        )
    except Exception as e:
        logging.error(f"Error exporting table: {e}")
        return jsonify({"error": "Failed to export table"}), 500

@app.route('/post_exp/3d-graph')
def graph_page():
    return render_template('3d_graph.html')

@app.route("/generate-3d-graph", methods=["POST"])
def generate_3d_graph():
    # 1) Parse request
    params      = [p.strip() for p in request.json["params"]]
    color_param = request.json.get("colorParam")
    if color_param:
        color_param = color_param.strip()

    x_key, y_key = params[0], params[1]
    z_key        = params[2] if len(params) > 2 else None
    c_key        = color_param

    x_raw, y_raw, z_raw, c_raw = [], [], [], []

    # 2) Collect data
    dataset = read_json(DATA_TABLE_PATH)
    for curves in dataset.values():
        for curve in curves.values():
            row = {k.strip(): v for k, v in curve.items()}
            if x_key not in row or y_key not in row or (z_key and z_key not in row):
                continue
            x_raw.append(row[x_key])
            y_raw.append(row[y_key])
            if z_key:
                z_raw.append(row[z_key])
            if c_key and c_key in row:
                c_raw.append(row[c_key])

    # 3) Convert to proper types
    def to_series(raw, key):
        if key == "Date and time measurement":
            # Try parsing a variety of common formats
            return pd.to_datetime(
                raw,
                infer_datetime_format=True,
                errors="coerce"  # bad parses become NaT
            )
        else:
            # Numeric: coerce non‐numeric to NaN
            return pd.to_numeric(raw, errors="coerce")

    data = {
        x_key: to_series(x_raw, x_key),
        y_key: to_series(y_raw, y_key),
    }
    if z_key:
        data[z_key] = to_series(z_raw, z_key)
    if c_key:
        data[c_key] = to_series(c_raw, c_key)

    df = pd.DataFrame(data).dropna(subset=[x_key, y_key] + ([z_key] if z_key else []))

    # 4) Plot
    labels = {x_key: x_key, y_key: y_key}
    if z_key:
        labels[z_key] = z_key
    if c_key:
        labels[c_key] = c_key

    if z_key:
        fig = px.scatter_3d(
            df,
            x=x_key, y=y_key, z=z_key,
            color=c_key,
            labels=labels,
            title=f"3D Graph of {x_key}, {y_key}, {z_key}"
        )
        # If your x/y/z include dates, tell Plotly to treat them as dates:
        layout_updates = {}
        if x_key == "Date and time measurement":
            layout_updates["scene.xaxis"] = dict(type="date")
        if y_key == "Date and time measurement":
            layout_updates["scene.yaxis"] = dict(type="date")
        if z_key == "Date and time measurement":
            layout_updates["scene.zaxis"] = dict(type="date")
        if layout_updates:
            fig.update_layout(**layout_updates)

    else:
        # 2D scatter — Plotly auto‐detects datetime series
        fig = px.scatter(
            df,
            x=x_key, y=y_key,
            color=c_key,
            labels=labels,
            title=f"2D Graph of {x_key} vs {y_key}"
        )

    # 5) Return JSON
    return jsonify(pio.to_json(fig))

@app.route('/download-results')
def download_results():
    try:
        if not os.path.exists(DEMO_JSON_PATH):
            return "No results found.", 404
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        download_name = f"apace_results_{timestamp}.json"
        return send_file(
            DEMO_JSON_PATH,
            mimetype='application/json',
            as_attachment=True,
            download_name=download_name
        )
    except Exception as e:
        logging.error(f"Error in download_results: {e}")
        return jsonify({"error": "Failed to download results"}), 500

@app.route('/exit', methods=['GET'])
def exit():
    try:
        if not os.path.exists(DEMO_JSON_PATH):
            return "No results found.", 404
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        download_name = f"apace_results_{timestamp}.json"
        response = send_file(
            DEMO_JSON_PATH,
            mimetype='application/json',
            as_attachment=True,
            download_name=download_name
        )
        def shutdown_server():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                logging.warning("Server shutdown not supported in this environment")
            else:
                func()
        threading.Timer(1.0, shutdown_server).start()
        return response
    except Exception as e:
        logging.error(f"Error in exit: {e}")
        return jsonify({"error": "Failed to exit and download results"}), 500

def clear_json_file():
    """Clear JSON files on application shutdown."""
    for path in [UPLOADED_FILES_PATH, UPLOADED_FOLDER_PATH, PASS_GRAPHS_PATH, FAIL_GRAPHS_PATH, DEMO_JSON_PATH, DATA_TABLE_PATH]:
        write_json(path, {})

atexit.register(clear_json_file)

if __name__ == '__main__':
    app.run(debug=True)