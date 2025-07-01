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
import statistics
import pandas as pd
import plotly.express as px
import ast

def find_opt(args):
    weight,All_combs_details = args
    #print(weight)
    Alg_comb_index = list(All_combs_details.keys())  
    user_utility = []
    for single_combo in Alg_comb_index:  
        if weight == .0:
            user_utility.append(-All_combs_details[single_combo]["Mean Square Error"] )
        elif weight == 1.0: #choose min NWMSE
            user_utility.append(All_combs_details[single_combo]["Success Rate"] - 0.00000001* All_combs_details[single_combo]["Mean Square Error"] )
        else:
            user_utility.append(All_combs_details[single_combo]["Success Rate"] - (1-weight)/weight* All_combs_details[single_combo]["Mean Square Error"] ) #relation 2
        #user_utility.append(weight*All_combs_details[single_combo]["Success Rate"] - (1-weight)*All_combs_details[single_combo]["Relative Cumulative Error"] ) #relation 1
    optimal_index = np.argmax( np.array(user_utility))
    #print(user_utility[optimal_index])

    
    return str(optimal_index) 
    


if __name__ == '__main__':


    with open('All Comb info.json', 'r', encoding='utf-8') as file:
        all_combs_details = json.load(file)


    all_SR = []
    all_NWMSE = []

    all_combs = list(all_combs_details.keys() )
    for comb in all_combs:
        all_SR.append( all_combs_details[comb]["Success Rate"]  )
        all_NWMSE.append( all_combs_details[comb]["Mean Square Error"]  )

    SR_mean = sum(all_SR)/len(all_SR)
    SR_std =  statistics.pstdev(all_SR)

    NWMSE_mean = sum(all_NWMSE)/len(all_NWMSE)
    NWMSE_std =  statistics.pstdev(all_NWMSE)


    print('The ideal point is :', [ (1 - SR_mean)/SR_std ] , ( 0 - NWMSE_mean)/NWMSE_std  )
    for comb in all_combs:

        all_combs_details[comb]["Success Rate"] = ( all_combs_details[comb]["Success Rate"] - SR_mean)/SR_std

        all_combs_details[comb]["Mean Square Error"] = ( all_combs_details[comb]["Mean Square Error"] - NWMSE_mean)/NWMSE_std




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
    fig_buffer.write_html('algorithm_performance_result_normalized.html')
    fig_buffer.show()
    df.to_csv('output_normalized.csv', index=True, encoding='utf-8-sig')


    with open('All Comb info_normalized.json', 'w') as json_file:
        json.dump(all_combs_details, json_file,indent=4,  ensure_ascii=False)


    select_combs_details = {}
    weight_index = 0 #recored index for weight
    # selected_alg_details = {}
    args = (1.,all_combs_details)
    selected_index = find_opt(args)
    select_combs_details[weight_index] = {}
    select_combs_details[weight_index]['Success Weight'] = 1. 
    select_combs_details[weight_index]['Baseline Fitting Algorithms'] = all_combs_details[selected_index]['Baseline Fitting Algorithms']
    select_combs_details[weight_index]['Ratio For Peak'] = all_combs_details[selected_index]['Ratio For Peak']
    select_combs_details[weight_index]['CPD Search Model'] = all_combs_details[selected_index]['CPD Search Model']
    select_combs_details[weight_index]['CPD Cost Function'] = all_combs_details[selected_index]['CPD Cost Function']
    select_combs_details[weight_index]['Success Rate'] = all_combs_details[selected_index]['Success Rate']    
    select_combs_details[weight_index]['Mean Square Error'] = all_combs_details[selected_index]['Mean Square Error']    
        


    
    weight_index = 0 

    args = [ (i/100. ,all_combs_details ) for i in range(101 ) ]

    num_cpus = cpu_count()

    with Pool(processes=num_cpus) as pool:
        results = pool.map(find_opt, args) 

    # print(results)
    for result in results: # reuslt:file level ; results:folder level
        selected_index =  result
        select_combs_details[weight_index] = {}
        select_combs_details[weight_index]['Success Weight'] = 1. 
        select_combs_details[weight_index]['Baseline Fitting Algorithms'] = all_combs_details[selected_index]['Baseline Fitting Algorithms']
        select_combs_details[weight_index]['Ratio For Peak'] = all_combs_details[selected_index]['Ratio For Peak']
        select_combs_details[weight_index]['CPD Search Model'] = all_combs_details[selected_index]['CPD Search Model']
        select_combs_details[weight_index]['CPD Cost Function'] = all_combs_details[selected_index]['CPD Cost Function']
        select_combs_details[weight_index]['Success Rate'] = all_combs_details[selected_index]['Success Rate']    
        select_combs_details[weight_index]['Mean Square Error'] = all_combs_details[selected_index]['Mean Square Error']    
        weight_index += 1

        # plt.scatter( all_combs_details[selected_index]['Success Rate'],all_combs_details[selected_index]['Relative Cumulative Error'],color='red', s=20 )


    


    # # Labels and title
    # handles, labels = ax.get_legend_handles_labels()
    # unique_legend = OrderedDict(zip(labels, handles))  # 使用 OrderedDict 去重

    # # 更新图例
    # ax.legend(unique_legend.values(), unique_legend.keys(), loc='best')
    # plt.ylim(0,0.3)
    # plt.title('Algorithm Set Performance')

    # plt.legend()

    # plt.grid(True)
    # plt.show()
    # os.chdir('../')


    # with open('All Comb info_alt.json', 'w') as json_file:
    #     json.dump(all_combs_details, json_file,indent=4,  ensure_ascii=False)

    with open('Algorithm Setting.json', 'w') as json_file:
        json.dump(select_combs_details, json_file,indent=4,  ensure_ascii=False)


