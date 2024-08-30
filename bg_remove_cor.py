#脑肠亚型分类代码BG
#Written by Raven
from __future__ import division
#########################################################
#                                  实验目的：去除协变量（脑肠）                                #
#                                        written by Raven                                         #
#########################################################






#导入各种包
#####
import numpy as np
import matplotlib.pyplot as plt  #加载画图的包
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
from pygam import LogisticGAM
import pandas as pd
import statsmodels.api as sm
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####

# 载入数据远程CPU
#####
random_seed = 42
np.random.seed(random_seed)
dataset_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/renkouxue_all.csv',encoding='gb18030')
dataset_fc_matrix.dropna(inplace=True)
dataset_fc_matrix.head()
print(dataset_fc_matrix)
print(dataset_fc_matrix.columns.tolist())  #列出每个feature的名字
healthy_str = 'HC'
patient_str = 'SZ'
print('Number of paticipants = %d' % dataset_fc_matrix.shape[0])  #该元组的第一个元素代表行数，第二个元素代表列数
A_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/regenus.csv',encoding='gb18030')
A_fc_matrix.dropna(inplace=True)
A_fc_matrix.head()
print('相对丰度矩阵', A_fc_matrix)
B_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/alpha.csv',encoding='gb18030')
B_fc_matrix.dropna(inplace=True)
B_fc_matrix.head()
print('alpha', B_fc_matrix)
C_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/ALFF.csv',encoding='gb18030')
C_fc_matrix.dropna(inplace=True)
C_fc_matrix.head()
print('ALFF', C_fc_matrix)
D_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/ReHo.csv',encoding='gb18030')
D_fc_matrix.dropna(inplace=True)
D_fc_matrix.head()
print('REHO', D_fc_matrix)
E_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GMV.csv',encoding='gb18030')
E_fc_matrix.dropna(inplace=True)
E_fc_matrix.head()
print('GMV', E_fc_matrix)
F_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/WMV.csv',encoding='gb18030')
F_fc_matrix.dropna(inplace=True)
F_fc_matrix.head()
print('WMV', F_fc_matrix)
demographic = pd.read_csv("/mnt/disk1/wyr_data/bg_net_sz/renkouxue_98.csv",encoding='gb18030')
demographic.dropna(inplace=True)
demographic.head()
print('人口学信息', demographic)
#####



# 对原始数据去除协变量——GLM_ab
#####
merged_df = pd.concat([A_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=A_fc_matrix.columns)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
A = pd.DataFrame(index=merged_df.index, columns=A_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in A_fc_matrix.columns:
    y = A_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    A[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(A)
A.to_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/regenus.csv',header=True, index=False)
#####