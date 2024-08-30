#脑肠亚型分类代码-肠
#Written by Raven
from __future__ import division

import math

#########################################################
#                               实验目的：构建个体肠网络和热图                                #
#                                        written by Raven                                         #
#########################################################






#导入各种包
#####
import numpy as np
import matplotlib.pyplot as plt  #加载画图的包
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr
import pandas as pd
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####





# # 肠网络构建
# import numpy as np
# import pandas as pd
# import os
# data = pd.read_csv('D:\AAARavencode\Practice\pycharm\AAAcluster_final\gut_genus.csv', encoding='gb18030',
#                    encoding_errors='ignore')
# num_people, num_features = data.shape
# output_path = 'G:/gut_brainMRI/gut_result'
# # 确保目标路径存在
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# # 初始化相关系数矩阵
# num_features = data.shape[0]
# correlation_matrix = np.zeros((num_features, num_features))
# for i in range(num_people):
#     person_data = data.iloc[i:i + 1].values[0]
#     for j in range(num_features):
#         mean_x = person_data[j]
#         mean_y = person_data[j+1]
#         for t in range(i):
#             up=sum((person_data[t]-mean_x) * (person_data[t+1]-mean_y))
#             down_x = np.sqrt(sum((person_data[t] - mean_x) ** 2))
#             down_y = np.sqrt(sum((person_data[t] - mean_y) ** 2))
#             correlation_matrix = up / (down_x * down_y)
#     file_name = f'gut_cor{i + 1}.txt'
#     file_path = os.path.join(output_path, file_name)
#     np.savetxt(file_path, correlation_matrix, fmt='%.6f', delimiter='\t')




#最终版
import numpy as np
import pandas as pd
import os
import seaborn as sns
data = pd.read_csv('G:\Gut316_gutonly\\regenus.csv', encoding='gb18030',
                   encoding_errors='ignore')
num_people, num_features = data.shape
output_path = 'G:\Gut316_gutonly\\gutnet'
# 确保目标路径存在
if not os.path.exists(output_path):
    os.makedirs(output_path)
epsilon = 0.001
data_with_epsilon = data + epsilon # 添加常数
mean = np.mean(data_with_epsilon).values
correlation_matrix = np.zeros((num_features, num_features))
def variance_to_similarity(variance, variances=None):
    # 使用线性逆变换将方差映射到[0, 1]范围
    similarity = 1.0 - (variance / np.max(variances))
    return similarity
for t in range(num_people): # 找到个人
    person_data=data_with_epsilon.iloc[t:t+1].values
    for i in range(num_features): #X
        for j in range(num_features): #Y
            person_data_i = person_data[:,i]
            person_data_j = person_data[:,j]
            log = np.log(person_data_i / person_data_j)
            log_mean = np.log(mean[i] / mean[j])
            num = np.sum((log-log_mean)**2)
            correlation_matrix[i,j] = num
    correlation_matrix = np.sqrt(correlation_matrix)
    min_value = np.min(correlation_matrix)
    max_value = np.max(correlation_matrix)
    normalized_data =1- (correlation_matrix - min_value) / (max_value - min_value)
    sns.heatmap(normalized_data,cmap='magma')
    plt.xlabel(f'Correlation Matrix of No.{t+1} Participants')
    plt.savefig(f'G:\Gut316_gutonly\\gutnet\heatmap\HeatMap of No. {t+1} Participants.tiff')
    plt.show()
    print(f'第{t+1}个被试矩阵')
    print(normalized_data)
    file_name = f'gut_cor{t + 1}.txt'
    file_path = os.path.join(output_path, file_name)
    np.savetxt(file_path, normalized_data, fmt='%.6f', delimiter='\t')





# import numpy as np
# data = np.array([[10, 20, 30, 40],
#                  [20, 30, 40, 50],
#                  [5, 15, 25, 35],
#                  [15, 25, 35, 45]])
#
# # 计算Pearson相关性矩阵
# def pearson_correlation_matrix(data):
#     num_samples, num_microbes = data.shape
#     correlation_matrix = np.zeros((num_microbes, num_microbes))
#
#     for i in range(num_microbes):
#         for j in range(num_microbes):
#             # 计算Pearson相关性系数
#             r, _ = np.corrcoef(data[:, i], data[:, j])
#             correlation_matrix[i, j] = r[0]
#
#     return correlation_matrix
#
# correlation_matrix = pearson_correlation_matrix(data)
#
# # 设置显著性阈值
# threshold = 0.5
#
# # 构建微生物网络
# microbiome_network = {
#     i: [] for i in range(data.shape[0])
# }
#
# for i in range(data.shape[0]):
#     for j in range(data.shape[0]):
#         if i != j and np.abs(correlation_matrix[i, j]) > threshold:
#             microbiome_network[i].append(j)
#
# print("Microbiome Network:")
# for node, neighbors in microbiome_network.items():
#     print(f"Node {node} is connected to: {neighbors}")