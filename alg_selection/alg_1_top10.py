#find alternative algorithm set for default one
import json
import os
import re
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count,freeze_support
import matplotlib.cm as cm
from collections import OrderedDict
import plotly.express as px
from collections import defaultdict
from scipy.stats import linregress
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess


def drop_Diff_cal( Alg_Data ):

    keys2exclude = ['Algorithms: ', 'Total Number of Curves', 'Total Number of Fails']
    # R_success = 1- Alg_Data['Total Number of Fails']/Alg_Data['Total Number of Curves']
    #folders = [key for key in Alg_Data .keys() if key not in keys2exclude ]

    # num_folder = len(Alg_Data)
    # error = 0 #accumulated error
    # num_curve2cal_error = 0   #number of curves for error average calculation

    # Alg_total_curve_num = 0
    # Alg_curve_right_num = 0
    all_MSE = []

    all_SRs = []
    for folder_index in range( len(list(Alg_Data.keys())) - 3 ):
        folder = str(folder_index + 1)
        
        Raw_Peak = Alg_Data[folder][ "Raw Data Peak "]
        Net_Peak = Alg_Data[folder ]["Net Peak "]
        # Alg_total_curve_num += len(Raw_Peak)
        # for k in range(len(Raw_Peak)):
        #     if Raw_Peak[k] >0 and Net_Peak[k] > 0 :
        #         Alg_curve_right_num += 1

        #save info of peaks that passed

        # Raw_Peak_success = []
        Net_Peak_success = []
        fake_x = []

        num_peaks = len(Net_Peak)
        for i in range(num_peaks):
            if Net_Peak[i] > 0 and not math.isnan(Net_Peak[i] ):
                fake_x.append(i)
                Net_Peak_success.append(Net_Peak[i])
        all_SRs.append(len(Net_Peak_success)/num_peaks  )
        

        if len(Net_Peak_success) > 1:
            lowess_smoothed = lowess(
                endog=Net_Peak_success,     # y 值
                exog=fake_x,            # x 值
                frac= 0.3 ,          # 邻域窗口大小（可根据实际情况多尝试不同值）
                it=3,              # 迭代次数
                return_sorted=True # 返回排序后的 (x, y) 对
            )[:,1]
            mse = mean_squared_error(Net_Peak_success, lowess_smoothed)/( max(Net_Peak_success)-min(Net_Peak_success)  )**2
            all_MSE.append(mse)

    if not all_MSE:
        all_MSE.append(1)


    return sum(all_SRs)/len(all_SRs),  sum(all_MSE)/len(all_MSE)


    # if num_curve2cal_error > 0:
    #     return Alg_curve_right_num/Alg_total_curve_num, error/num_curve2cal_error
    # else:
    #     return Alg_curve_right_num/Alg_total_curve_num, 99.




def cal_alg_results( file_name  ):


    with open(file_name, 'r', encoding='utf-8') as file:

        data_baselines= json.load(file)

    algsets_1 = list(data_baselines.keys())
    Alg_comb_details = {}


    for alg_index in range( len( algsets_1) ): 
        alg_1 = algsets_1[alg_index]
        Alg_comb_details[alg_1] = {}
        SR_temp, MSE_temp = drop_Diff_cal (data_baselines[alg_1])
        Alg_comb_details[alg_1]['Success Rate'] = SR_temp
        Alg_comb_details[alg_1]['Mean Square Error'] = MSE_temp

    return Alg_comb_details



def find_opt(args):
    weight,All_combs_details = args
    #print(weight)
    Alg_comb_index = list(All_combs_details.keys())  
    user_utility = []
    for single_combo in Alg_comb_index:  
        if weight == .0:
            user_utility.append(-All_combs_details[single_combo]["Relative Cumulative Error"] )
        else:
            user_utility.append(All_combs_details[single_combo]["Success Rate"] - (1-weight)/weight* All_combs_details[single_combo]["Relative Cumulative Error"] ) #relation 2
        #user_utility.append(weight*All_combs_details[single_combo]["Success Rate"] - (1-weight)*All_combs_details[single_combo]["Relative Cumulative Error"] ) #relation 1
    optimal_index = np.argmax( np.array(user_utility))
    print(user_utility[optimal_index])

    
    return optimal_index 
    


if __name__ == '__main__':
    threshold = 0.65

    # alg_comb_list = os.listdir('database')
    # os.chdir('database')
    # print(alg_comb_list)

    # os.chdir('result')
    alg_comb_list = [

        'peak_info_result_50_'+str(threshold) +'_BottomUp_rank.json',
        'peak_info_result_50_'+str(threshold) +'_Dynp_rank.json'

    ]


    all_combs_details = {}
    comb_index = 0

    pattern = pattern = r'peak_info_result_\d+_(\d+\.\d+)_(\w+)_(\w+)\.json'#extract info from file name 
    color = [ ]
    cmap = plt.cm.get_cmap("tab20", 10)

    added_length_labels = []


    for i in range(len(alg_comb_list)) :
        comb_name =  alg_comb_list[i] 

        match = re.search(pattern, comb_name)

        one_comb_set_results = cal_alg_results ( comb_name )
        combsinset = list(one_comb_set_results.keys()) #combs in one file
        print(comb_name,len(list(one_comb_set_results.keys())))
        for single_comb in combsinset:

            tuple_single_comb = eval(single_comb)
            length_single_comb = len(tuple_single_comb)
            SR = one_comb_set_results[single_comb]['Success Rate']
            MSE = one_comb_set_results[single_comb]['Mean Square Error']


            all_combs_details[comb_index] = {}
            all_combs_details[comb_index]['Baseline Fitting Algorithms'] = single_comb
            all_combs_details[comb_index]['Ratio For Peak'] = match.group(1)
            all_combs_details[comb_index]['CPD Search Model'] = match.group(2)
            all_combs_details[comb_index]['CPD Cost Function'] = match.group(3)
            all_combs_details[comb_index]['Success Rate'] = SR
            all_combs_details[comb_index]['Mean Square Error'] = MSE

            comb_index += 1
            # if length_single_comb != 1:
            #     continue

    df = pd.DataFrame.from_dict(all_combs_details, orient='index')

    # 合并标签：将 CPD Search Model 和 CPD Cost Function 连接起来
    df['Label'] = df['CPD Search Model'].astype(str) + ' ' + df['CPD Cost Function'].astype(str)

    # 利用 Plotly Express 绘制 3D 散点图
    # x轴为 'Success Rate', y轴为 'R Square Buffer', z轴为 'R Square Target'
    fig_buffer = px.scatter(
        df,
        x='Success Rate',
        y='Mean Square Error',
        color='Label',  # 使用合并后的标签进行着色
        hover_data=['Baseline Fitting Algorithms'],  # 鼠标悬停时显示更多信息
        title='Success Rate vs Mean Square Error'
    )

    # 显示交互式图表
    # 保存为 HTML 文件
    fig_buffer.write_html('algorithm_performance.html')
    fig_buffer.show()


    select_combs_details = {}



    with open('All Comb info_single.json', 'w') as json_file:
        json.dump(all_combs_details, json_file,indent=4,  ensure_ascii=False)

    with open('Algorithm Setting_single.json', 'w') as json_file:
        json.dump(select_combs_details, json_file,indent=4,  ensure_ascii=False)



    # 首先我们需要把所有条目根据 (CPD Search Model, CPD Cost Function) 分组
    grouped_details = defaultdict(list)

    for key, value in all_combs_details.items():
        model = value['CPD Search Model']
        cost = value['CPD Cost Function']
        sr = value['Success Rate']
        mse = value['Mean Square Error']
        algs = value['Baseline Fitting Algorithms']

        # 将需要排序的关键信息一并存放，方便后续排序
        grouped_details[(model, cost)].append((key, sr, mse, algs))

    num_top = 5
    # 定义想要遍历的顺序
    categories_order = [

        ('BottomUp', 'rank'),

        ('Dynp', 'rank')
    ]
    for category in categories_order:
        if category not in grouped_details:
            continue

        items = grouped_details[category]
        
        # 1. 按 (SR 降序, MSE 升序) 排序后取前10
        top_SR = sorted(
            items,
            key=lambda x: (-x[1], x[2]) 
        )[:num_top]
        

        top_MSE = sorted(
            items,
            key=lambda x: (x[2],-x[1]) 
        )[:num_top]

        union_set = set(top_SR + top_MSE)
        

        union_list = sorted(union_set, key=lambda x: (-x[1], x[2]))
        
        # 输出该类的结果
        print(f"{category[0]}-{category[1]})-{len(union_list)} algorithms")
        for item in union_list:
            k, sr,mse, algs = item
            # 这里演示只打印算 baseline fitting 的组合，可以根据需要修改
            print('\''+algs[2:-3]+'\',')
        print("-" * 60)

