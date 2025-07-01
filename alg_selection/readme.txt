Steps for algorithm selection

Preparation:
1. Collect time-series data in different folders, and name after 1,2,3...
2. Estimate the noise level and threshold for each folders and modify the ' threshold ',  'noise_levels' and 'num_folders' in CPD.py file
3. Run file_path.py to obtain file names and accelerate the following steps


Selection:
1: Run CPD.py to obtain CPs
2. Run Alg_cal.py for 34 baselines for each CPD combination within all signals
3. Run Alg_select_1.py and get peak heights for each signals
4. Run alg_1_top10.py.  Get performance of every baseline fitting algorithm and selection the top 10 for combination
5. Modify the ' fit_alg_all' in Alg_select_10_1core.py based on the output of last step/
6. Run  Alg_select_10_1core.py and get peak heights for each signals
7. Run alt_10_result.py and get peak heights for each signals
8. Run Alg_setting_normalized.py to obtain the selection results based on 0-1.0 weight of SR
9. Replace the 'Algorithm Setting.json' in A-PACE with the one from last step
