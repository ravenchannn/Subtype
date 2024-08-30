#脑肠亚型分类代码-BG
#Written by Raven
from __future__ import division

import random

import matplotlib.pyplot as plt  # 加载画图的包
# 导入各种包
#####
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#########################################################
#                             实验目的：个体脑-肠网络打标与p计算                            #
#                                        written by Raven                                         #
#########################################################
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####


import pandas as pd
import numpy as np
#2类打标
# 读取CSV文件
label = pd.read_csv("/mnt/disk1/wyr/bg_sz/data_driven/Train/Final_cluster_label.csv", encoding='gb18030', header=None)  # 替换成您想要的标签
# 添加全部标签
label_nc = np.full(43,2,dtype=int) #97/26
label_nc = label_nc.reshape(-1,1)
label_nc = pd.DataFrame(label_nc)
label_all = label_nc._append(label)
label_all = np.array(label_all)
label_all = pd.DataFrame(label_all)
# label_all.to_csv('/mnt/disk1/wyr/result_brain_net/all_final_label.csv',header=None,index=False)
# 读取要加标签的文件
input_file = "/mnt/disk1/wyr_data/bg_net_sz/Train/set.csv"  # 替换成您的输入文件名
output_file = '/mnt/disk1/wyr/bg_sz/differ/Train/features.csv'  # 替换成您的输出文件名
# 读取CSV文件到DataFrame
fc_matrix = pd.read_csv(input_file)
# 在DataFrame的最前列添加标签列
fc_matrix.insert(0, 'id', label_all)
# 将修改后的DataFrame保存为新的CSV文件
fc_matrix.to_csv(output_file, index=False)
print('ok')




# p值计算
# fMRI
# 全显著
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
cluster_demographic = pd.read_csv('/mnt/disk1/wyr/bg_sz/differ/featuresall.csv')
cluster_demographic.reset_index(drop=True, inplace=True)
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
result_df_all = pd.DataFrame(index=variable_columns_all, columns=['Kruskal-Wallis Statistic', 'P-value_all'])
for variable in variable_columns_all:
    group_data = [cluster_demographic[cluster_demographic[category_column_all] == category][variable]
                  for category in cluster_demographic[category_column_all].unique()]
    statistic, p_value = kruskal(*group_data)
    result_df_all.loc[variable, 'Kruskal-Wallis Statistic'] = statistic
    result_df_all.loc[variable, 'P-value_all'] = p_value
# 进行 FDR校正
p_values_fdr = multipletests(result_df_all['P-value_all'], alpha=0.05, method='fdr_bh')[1]
result_df_all['P-value (FDR)'] = p_values_fdr
# 进行 Bonferroni 校正
p_values_b = multipletests(result_df_all['P-value_all'], alpha=0.05, method='bonferroni')[1]
result_df_all['P-value (Bonferroni)'] = p_values_b
print('P-Value of all', result_df_all)
result_df_all.to_csv('/mnt/disk1/wyr/bg_sz/differ/p_value2.csv')
grouped_data = cluster_demographic.groupby('id')
means = grouped_data.mean()
std = grouped_data.std()
sem = std / np.sqrt(len(grouped_data))
print('means',means)
means.to_csv('/mnt/disk1/wyr/bg_sz/differ/means.csv', index=False)
#####

from itertools import combinations
from scipy.stats import kruskal
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
# 存储结果的DataFrame
result_df = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# 对每两组进行两两比较
for group1, group2 in combinations(cluster_demographic[category_column_all].unique(), 2):
    for variable in variable_columns_all:
        data_group1 = cluster_demographic.loc[cluster_demographic[category_column_all] == group1, variable]
        data_group2 = cluster_demographic.loc[cluster_demographic[category_column_all] == group2, variable]
        # 执行 KW 检验
        stat, p_value = kruskal(data_group1, data_group2)
        # 将结果添加到DataFrame
        row_data = [variable, group1, group2, stat, p_value]
        result_df = pd.concat([result_df, pd.DataFrame([row_data], columns=result_df.columns)])
# 进行 FDR校正
p_values_fdr = multipletests(result_df['P-value'], alpha=0.05, method='fdr_bh')[1]
result_df['P-value (FDR)'] = p_values_fdr
# 进行 Bonferroni 校正
p_values_b = multipletests(result_df['P-value'], alpha=0.05, method='bonferroni')[1]
result_df['P-value (Bonferroni)'] = p_values_b
print(result_df)
result_df.to_csv('/mnt/disk1/wyr/bg_sz/differ/p_values3.csv',index=False)
