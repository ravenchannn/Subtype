#脑肠亚型分类代码-脑
#Written by Raven
from __future__ import division
#########################################################
#                                    实验目的：去除协变量（脑）                                 #
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






#载入数据远程CPU
#####
random_seed = 42
np.random.seed(random_seed)
dataset_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/renkouxue_all.csv',encoding='gb18030')
dataset_fc_matrix.dropna(inplace=True)
dataset_fc_matrix.head()
print(dataset_fc_matrix)
print(dataset_fc_matrix.columns.tolist())  #列出每个feature的名字
healthy_str = 'HC'
patient_str = 'SZ'
print('Number of paticipants = %d' % dataset_fc_matrix.shape[0])  #该元组的第一个元素代表行数，第二个元素代表列数
A_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/brain/ALFF_all.csv',encoding='gb18030')
A_fc_matrix.dropna(inplace=True)
A_fc_matrix.head()
print('ALFF矩阵', A_fc_matrix)
B_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/brain/ReHo_all.csv',encoding='gb18030')
B_fc_matrix.dropna(inplace=True)
B_fc_matrix.head()
print('ReHo矩阵', B_fc_matrix)
C_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/brain/DC_all.csv',encoding='gb18030')
C_fc_matrix.dropna(inplace=True)
C_fc_matrix.head()
print('DC矩阵', C_fc_matrix)
D_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/brain/GMV_all.csv',encoding='gb18030')
D_fc_matrix.dropna(inplace=True)
D_fc_matrix.head()
print('GMV矩阵', D_fc_matrix)
E_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_brain550/brain/WMV_all.csv',encoding='gb18030')
E_fc_matrix.dropna(inplace=True)
E_fc_matrix.head()
print('WMV矩阵', E_fc_matrix)
F_fc_matrix = pd.read_csv("/mnt/disk1/wyr_data/SZ_brain550/network/BC_all.csv",encoding='gb18030')
F_fc_matrix.dropna(inplace=True)
F_fc_matrix.head()
print('BC矩阵', F_fc_matrix)
G_fc_matrix = pd.read_csv("/mnt/disk1/wyr_data/SZ_brain550/network/NE_all.csv",encoding='gb18030')
G_fc_matrix.dropna(inplace=True)
G_fc_matrix.head()
print('NE矩阵', G_fc_matrix)
demographic = pd.read_csv("/mnt/disk1/wyr_data/SZ_brain550/renkouxue_all.csv",encoding='gb18030')
demographic.dropna(inplace=True)
demographic.head()
print('人口学信息', demographic)
#####





# 对原始数据去除协变量——GLM_ALFF
#####
merged_df = pd.concat([A_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=A_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
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
A.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/ALFF.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_ReHo
#####
merged_df = pd.concat([B_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=B_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
B = pd.DataFrame(index=merged_df.index, columns=B_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in B_fc_matrix.columns:
    y = B_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    B[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(B)
B.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/ReHo.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_DC
#####
merged_df = pd.concat([C_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=C_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
C = pd.DataFrame(index=merged_df.index, columns=C_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in C_fc_matrix.columns:
    y = C_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    C[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(C)
C.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/DC.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_GMV
#####
merged_df = pd.concat([D_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=D_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
D = pd.DataFrame(index=merged_df.index, columns=D_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in D_fc_matrix.columns:
    y = D_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    D[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(D)
D.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/GMV.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_WMV
#####
merged_df = pd.concat([E_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=E_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
E = pd.DataFrame(index=merged_df.index, columns=E_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in E_fc_matrix.columns:
    y = E_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    E[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(E)
E.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/WMV.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_BC
#####
merged_df = pd.concat([F_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=F_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
F = pd.DataFrame(index=merged_df.index, columns=F_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in F_fc_matrix.columns:
    y = F_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    F[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(F)
F.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/BC.csv',header=True, index=False)
#####





# 对原始数据去除协变量——GLM_NE
#####
merged_df = pd.concat([G_fc_matrix, demographic], axis=1)
result_df = pd.DataFrame(index=merged_df.index, columns=G_fc_matrix.columns)
merged_df['education'] = pd.to_numeric(merged_df['education'].str.replace(' ', ''), errors='coerce')
merged_df['Height'] = pd.to_numeric(merged_df['Height'].str.replace(' ', ''), errors='coerce')
merged_df['Weight'] = pd.to_numeric(merged_df['Weight'].str.replace(' ', ''), errors='coerce')
merged_df['BMI'] = pd.to_numeric(merged_df['BMI'].str.replace(' ', ''), errors='coerce')
# 计算平均值
mean_education = merged_df['education'].mean()
mean_height = merged_df['Height'].mean()
mean_weight = merged_df['Weight'].mean()
mean_BMI = merged_df['BMI'].mean()
# 填充缺失值为平均值
merged_df['education'].fillna(mean_education, inplace=True)
merged_df['Height'].fillna(mean_height, inplace=True)
merged_df['Weight'].fillna(mean_weight, inplace=True)
merged_df['BMI'].fillna(mean_BMI, inplace=True)
# 处理无穷大值
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
G = pd.DataFrame(index=merged_df.index, columns=G_fc_matrix.columns)
selected_features = ['sex', 'age', 'education', 'Height', 'Weight', 'BMI']
for col in G_fc_matrix.columns:
    y = G_fc_matrix[col]
    X = merged_df[selected_features]  # Drop the current response variable from covariates
    # Add a constant term to the covariates
    X = sm.add_constant(X)
    # Fit a GLM model (assuming Gaussian family and identity link)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result_df = model.fit()
    # Save the residuals in the result_df with the correct index
    G[col] = result_df.resid_deviance
# Display the result DataFrame with residuals
print(G)
G.to_csv('/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/NE.csv',header=True, index=False)
#####


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = np.array(A_fc_matrix)
# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30,  alpha=0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# 绘制密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(data, shade=True)
plt.title('Density Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()



# 检查HC和SZ的匹配
from scipy.stats import sem, kruskal
label_nc = np.full(300,2,dtype=int)
label_nc = label_nc.reshape(-1,1)
label_nc = pd.DataFrame(label_nc)
label_sz = np.full(250,3,dtype=int)
label_sz = label_sz.reshape(-1,1)
label_sz = pd.DataFrame(label_sz)
label_all = label_nc._append(label_sz)
demographic['education'] = pd.to_numeric(demographic['education'].str.replace(' ', ''), errors='coerce')
demographic['Height'] = pd.to_numeric(demographic['Height'].str.replace(' ', ''), errors='coerce')
demographic['Weight'] = pd.to_numeric(demographic['Weight'].str.replace(' ', ''), errors='coerce')
demographic['BMI'] = pd.to_numeric(demographic['BMI'].str.replace(' ', ''), errors='coerce')
mean_education = demographic['education'].mean()
mean_height = demographic['Height'].mean()
mean_weight = demographic['Weight'].mean()
mean_BMI = demographic['BMI'].mean()
demographic['education'].fillna(mean_education, inplace=True)
demographic['Height'].fillna(mean_height, inplace=True)
demographic['Weight'].fillna(mean_weight, inplace=True)
demographic['BMI'].fillna(mean_BMI, inplace=True)
label_all.reset_index(drop=True, inplace=True)
demographic.reset_index(drop=True, inplace=True)
cluster_demographic = pd.concat([label_all,demographic],axis=1)
grouped_data = cluster_demographic.groupby(0)
category_column = cluster_demographic.columns[0]
variable_columns = cluster_demographic.columns[1:]
result_df = pd.DataFrame(index=variable_columns, columns=['Kruskal-Wallis Statistic', 'P-value'])
for variable in variable_columns:
    group_data = [cluster_demographic[cluster_demographic[category_column] == category][variable]
                  for category in cluster_demographic[category_column].unique()]
    statistic, p_value = kruskal(*group_data)
    result_df.loc[variable, 'Kruskal-Wallis Statistic'] = statistic
    result_df.loc[variable, 'P-value'] = p_value
print('P-Value of Brain Subtypes', result_df)
means = grouped_data.mean()
std = grouped_data.std()
sem = std / np.sqrt(len(grouped_data))
print('means',means)
print('std', std)
print('sem',sem)
result_df.to_csv('/mnt/disk1/wyr/result_brain_net/KW_test_HC_SZ.csv',index=False)
means.to_csv('/mnt/disk1/wyr/result_brain_net/means.csv', index=False)
std.to_csv('/mnt/disk1/wyr/result_brain_net/std.csv', index=False)
sem.to_csv('/mnt/disk1/wyr/result_brain_net/sem.csv', index=False)







# 矩阵去除协变量-删除此想法
# import numpy as np
# import statsmodels.api as sm
# np.random.seed(42)
# with open('/mnt/disk1/wyr_data/datasets_SZ_brain_net_label/0_average_matrix.txt', 'r') as file:
#     Y = file.read()
# X=pd.read_csv("/mnt/disk1/wyr/result_brain_net/Demographic/means_all.csv")
# matrix = np.fromstring(Y, sep=' ').reshape(90, 90)
# matrix_flatten = matrix.flatten()
# Y = matrix_flatten.reshape(-1,1)
# X = X[0:1]
# X = sm.add_constant(X)
# num_obs = matrix.shape[0] * matrix.shape[1]
# X = np.tile(X, (num_obs, 1))
# model = sm.GLM(Y, X, family=sm.families.Gaussian())
# result = model.fit()
# residuals = result.resid_deviance
# residuals_matrix = residuals.reshape(90, 90)
# # 现在 'residuals' 包含了去除协变量影响后的残差矩阵