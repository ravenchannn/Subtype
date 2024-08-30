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
#                          实验目的：个体脑-肠网络数据驱动模型1                           #
#                                        written by Raven                                         #
#########################################################
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####





#载入数据远程CPU
#####
random_seed = 42
np.random.seed(42)
np.random.seed(random_seed)
dataset_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/renkouxue_98.csv',encoding='gb18030')
dataset_fc_matrix.dropna(inplace=True)
dataset_fc_matrix.head()
print(dataset_fc_matrix)
print(dataset_fc_matrix.columns.tolist())  #列出每个feature的名字
healthy_str = 'HC'
patient_str = 'SZ'
print('Number of paticipants = %d' % dataset_fc_matrix.shape[0])  #该元组的第一个元素代表行数，第二个元素代表列数
C_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/ALFF.csv',encoding='gb18030')
print('ALFF', C_fc_matrix)
D_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/ReHo.csv',encoding='gb18030')
print('ReHo', D_fc_matrix)
B_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/WMV.csv',encoding='gb18030')
print('GMV', B_fc_matrix)
A_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/GMV.csv',encoding='gb18030')
print('GMV', A_fc_matrix)
E_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/regenus.csv',encoding='gb18030')
print('regenus', E_fc_matrix)
F_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/alpha.csv',encoding='gb18030')
print('GMV', F_fc_matrix)
#####





# 合并Brain&Gut
#####
Brain = pd.concat([A_fc_matrix, B_fc_matrix, C_fc_matrix, D_fc_matrix], axis=1)
Gut = pd.concat([E_fc_matrix,F_fc_matrix],axis=1)
Brain_1 = Brain.T
gut = Gut[55:]
brain = Brain_1.iloc[:,55:]
#####




# 合并Brain&Gut_NC
#####
gut2 = Gut[:55]
brain2 = Brain_1.iloc[:,:55]
#####





# 点乘融合+向量化
#####
result={}
results={}
for i in range(len(gut)):
    A = pd.DataFrame(brain.iloc[:, i].values.reshape(-1, 1))  # 将 Series 转换为 DataFrame
    B = pd.DataFrame(gut.iloc[i, :].values.reshape(1, -1))  # 将 Series 转换为 DataFrame
    results[i] = np.dot(A,B)
    result[i] = np.dot(A, B).flatten()  # 或者使用 .ravel() 方法
    save_path = "/mnt/disk1/wyr/bg_sz/FC/"
    file_name = f"SZ_{i+1}.txt"
    # 将矩阵转换为 DataFrame，并保存为 CSV 文件
    result[i]=pd.DataFrame(result[i])
    results[i]=pd.DataFrame(results[i])
    results[i].to_csv(save_path + file_name, sep='\t',header=None,index=False)
result = np.array(list(result.values()))
result = result.reshape(result.shape[0],-1)
result = pd.DataFrame(result)
result.to_csv('/mnt/disk1/wyr/bg_sz/BGmatrix.csv',header=None,index=False)
print(result)
#####





# 初始化
#####
data1=pd.read_csv("/mnt/disk1/wyr/bg_sz/BGmatrix.csv",header=None)
data2=pd.read_csv("/mnt/disk1/wyr/bg_sz/BGmatrix_HC.csv",header=None)
scaler = StandardScaler()
data1 = scaler.fit_transform(data1) # 标准化
data2 = scaler.fit_transform(data2) # 标准化
data1=pd.DataFrame(data1)
data2=pd.DataFrame(data2)
#####





# 划分TV
#####
# SZ
train_sets = []
validation_sets = []
grouped = data1.groupby(data1.index // 10)
for name, group in grouped:
    train_set = group.head(8)
    validation_set = group.tail(2)
    train_sets.append(train_set)
    validation_sets.append(validation_set)
train_df = pd.concat(train_sets)
train_df = train_df.iloc[:-2]
validation_df = pd.concat(validation_sets)
print("\n训练集:")
print(train_df)
print("\n验证集:")
print(validation_df)
# HC
# 初始化训练集和验证集的列表
training_sets = []
validate_sets = []
grouped1 = data2.groupby(data2.index // 10)
for name, group in grouped1:
    training_set = group.head(8)
    validate_set = group.tail(2)
    training_sets.append(training_set)
    validate_sets.append(validate_set)
training_df = pd.concat(training_sets)
training_df = training_df.iloc[:-2]
validate_df = pd.concat(validate_sets)
print("\n训练集:")
print(training_df)
print("\n验证集:")
print(validate_df)
ttttt = pd.concat([training_df,train_df], ignore_index=True)
vvvvv = pd.concat([validate_df,validation_df], ignore_index=True)
ttttt.to_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/Set.csv',header=None,index=False)
vvvvv.to_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/Set.csv',header=None,index=False)
#####





# 循环画热图
#####
data_folder = '/mnt/disk1/wyr_data/bg_net_sz/FC/'
save_folder = '/mnt/disk1/wyr_data/bg_net_sz/heatmap/'
txt_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
for file_path in txt_files:
    data = np.loadtxt(file_path)
    plt.figure(figsize=(8, 6))
    sns.heatmap(data,cmap='magma')
    plt.title(os.path.basename(file_path))  # 使用文件名作为标题
    file_name = os.path.basename(file_path)
    output_file = os.path.join(save_folder, os.path.splitext(file_name)[0] + '.tiff')
    plt.savefig(output_file, format='tiff')
    plt.show()
#####