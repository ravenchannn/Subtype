#脑肠亚型分类代码-肠
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
#########################################################
#                               实验目的：个体肠网络数据驱动模型                             #
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
dataset_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/demographic.csv',encoding='gb18030')
dataset_fc_matrix.dropna(inplace=True)
dataset_fc_matrix.head()
print(dataset_fc_matrix)
print(dataset_fc_matrix.columns.tolist())  #列出每个feature的名字
healthy_str = 'HC'
patient_str = 'SZ'
print('Number of paticipants = %d' % dataset_fc_matrix.shape[0])  #该元组的第一个元素代表行数，第二个元素代表列数
A_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/abgenus.csv',encoding='gb18030')
print('绝对丰度矩阵', A_fc_matrix)
B_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/DC.csv',encoding='gb18030')
print('DC矩阵', B_fc_matrix)
C_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/BC.csv',encoding='gb18030')
print('BC矩阵', C_fc_matrix)
D_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/NE.csv',encoding='gb18030')
print('NE矩阵', D_fc_matrix)
E_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/regenus.csv',encoding='gb18030')
print('regenus矩阵', E_fc_matrix)
F_fc_matrix = pd.read_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/alpha.csv',encoding='gb18030')
print('alpha矩阵', F_fc_matrix)
#####





# 点乘融合s+f+n
B_fc_matrix_1 = B_fc_matrix.T
D_fc_matrix_1 = D_fc_matrix.T
matrices = [A_fc_matrix,B_fc_matrix_1,C_fc_matrix, D_fc_matrix_1, E_fc_matrix]# 将多个矩阵存储在一个列表中
result = matrices[0]
for i in range(1, len(matrices)):
    result = np.dot(result, matrices[i])
result = np.concatenate((result, F_fc_matrix), axis=1) #1加列后hstack，0加载行后vstack
#####






# 初始化
#####
scaler = StandardScaler()
result = scaler.fit_transform(result) # 标准化
#####




#选取患者的大脑脑区
#####
data = result[26:]
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('患者', data) #取所需要的数据
#####





# # 选择训练集和验证集
# #####
# # SZ
# # 初始化训练集和验证集的列表
# result = dataset_fc_matrix
# data = result[123:]
# data.reset_index(drop=True, inplace=True)
# atad = result[0:123]
# atad.reset_index(drop=True, inplace=True)
# train_sets = []
# validation_sets = []
# grouped = data.groupby(data.index // 10)
# for name, group in grouped:
#     train_set = group.head(8)
#     validation_set = group.tail(2)
#     train_sets.append(train_set)
#     validation_sets.append(validation_set)
# train_df = pd.concat(train_sets)
# train_df = train_df.iloc[:-2]
# validation_df = pd.concat(validation_sets)
# print("\n训练集:")
# print(train_df)
# print("\n验证集:")
# print(validation_df)
# # HC
# # 初始化训练集和验证集的列表
# training_sets = []
# validate_sets = []
# grouped1 = atad.groupby(atad.index // 10)
# for name, group in grouped1:
#     training_set = group.head(8)
#     validate_set = group.tail(2)
#     training_sets.append(training_set)
#     validate_sets.append(validate_set)
# training_df = pd.concat(training_sets)
# training_df = training_df.iloc[:-2]
# validate_df = pd.concat(validate_sets)
# print("\n训练集:")
# print(training_df)
# print("\n验证集:")
# print(validate_df)
# ttttt = pd.concat([training_df,train_df], ignore_index=True)
# vvvvv = pd.concat([validate_df,validation_df], ignore_index=True)
# ttttt.to_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/Set.csv',header=None,index=False)
# vvvvv.to_csv('/mnt/disk1/wyr_data/SZ_gut316/Validate/Set.csv',header=None,index=False)
# #####





#ICA降维
#####
from sklearn.decomposition import FastICA
ICA = FastICA(n_components=2, random_state=42)
lowDmat=ICA.fit_transform(data)
scaler = StandardScaler()
lowDmat = scaler.fit_transform(lowDmat)
print('降维后的数据维度：',lowDmat.shape)
reconMat_1=ICA.inverse_transform(lowDmat) # s重构数据
print("重构后的数据维度：",reconMat_1.shape) # 重构数据维度
plt.show()
plt.scatter(lowDmat[:, 0], lowDmat[:, 1], color='#483D8B', alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decomposition using ICA')
plt.savefig('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/ICA_lower_dimensionality.tiff', dpi=300)
plt.show()
scaler = StandardScaler()
lowDmat = scaler.fit_transform(lowDmat)
print('降维后',lowDmat)
print('重构后',reconMat_1)
lowDmat = pd.DataFrame(lowDmat)
lowDmat.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/ICA_lower_dimensionality.csv',index=False,header=None)
lowDmat = np.array(lowDmat)
#####





#自编码器数据增强
#####
from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense
from keras.optimizers import Adam
# 构建深层聚类模型
input_dim = lowDmat.shape[1]  # 输入维度
latent_dim = 1  # 潜在空间维度
# 定义函数来创建和训练深度聚类模型
def train_deep_clustering_model(input_data):
    # 编码器
    inputs = Input(shape=(input_dim,))
    encoder = Dense(128, activation='relu')(inputs)  # 添加一个隐藏层
    encoder = Dense(64, activation='relu')(encoder)  # 添加一个隐藏层
    encoder = Dense(32, activation='relu')(encoder)  # 添加第二个隐藏层
    encoder = Dense(latent_dim, activation='relu')(encoder)
    # 解码器
    decoder = Dense(32, activation='relu')(encoder)  # 添加一个隐藏层
    decoder = Dense(64, activation='relu')(decoder)  # 添加第二个隐藏层
    decoder = Dense(128, activation='relu')(decoder)  # 添加第二个隐藏层
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    # 噪声定义
    noisy = lowDmat + 0.1 * np.random.randn(*lowDmat.shape)  # 0.1是噪声水平
    # 完整的深层聚类模型
    autoencoder = Model(inputs, decoder)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    # 训练深层聚类模型_调参
    autoencoder.fit(input_data, noisy, batch_size=128, epochs=500,  shuffle=True)
    # 提取编码器输出作为特征表示
    encoder_model = Model(inputs, encoder)
    aa = encoder_model.predict(input_data)
    return aa
# 重复运行模型十次并保存每次的特征表示
all_features = []
for i in range(10):
    print('第{}次循环'.format(i+1))
    features = train_deep_clustering_model(lowDmat)
    all_features.append(features)
features = np.mean(all_features, axis=0)
scaler = StandardScaler()
features = scaler.fit_transform(features)
features_df=pd.DataFrame(features)
features_df.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Autoencoder_results.csv', index=False, header=None)
#####





# 混合聚类
#####
# 导入各种包
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
# SC系数
clusters = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    kmeans_model = KMeans(n_clusters=t).fit(features)
    # 绘制轮廓系数与不同类簇数量的直观显示图
    sc_score = silhouette_score(features, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    print('K = %s, Silhouette Coefficient= %0.03f' %(t, sc_score))
sc_df = pd.DataFrame(list(zip(clusters, sc_scores)), columns=['Number of Clusters', 'Silhouette Coefficient'])
sc_df.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/silhouette_coefficients.csv', index=False)
plt.plot(clusters, sc_scores, color='#bf1e2e',marker='o')
plt.title('Silhouette Coefficient')
plt.xticks(range(2, 16, 1))
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Components_Silhouette Coefficient')
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Silhouette Coefficient.tiff",dpi=300)
plt.show()

# BIC计算
bic_gmm = []
for n_components in range(2, 16):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)
    bic_gmm.append(gmm.bic(features))
    print('K = %s, bic= %0.03f' % (n_components, gmm.bic(features)))
bic_df = pd.DataFrame(list(zip(clusters, bic_gmm)), columns=['Number of Clusters', 'BIC'])
bic_df.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/BIC.csv', index=False)
plt.plot(clusters, bic_gmm, color='#1d3557',marker='o')
plt.title('Cluster Components_BIC')
plt.xticks(range(2, 16, 1))
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Components_BIC')
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/bic_gmm.tiff",dpi=300)
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata
# 聚类簇数范围
features=pd.read_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Train/Autoencoder_results.csv',header=None)
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    kmeans_model = KMeans(n_clusters=t).fit(features)
    # 绘制轮廓系数与不同类簇数量的直观显示图
    sc_score = silhouette_score(features, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    print('K = %s, Silhouette Coefficient= %0.03f' %(t, sc_score))
bic_gmm = []
for n_components in range(2, 16):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)
    bic_gmm.append(gmm.bic(features))
    print('K = %s, bic= %0.03f' % (n_components, gmm.bic(features)))
z =sc_scores
y = bic_gmm
x = clusters
xi = np.linspace(min(x), max(x), 1000)
yi = np.linspace(min(y), max(y), 1000)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='RdBu_r', edgecolor='none')
ax.plot_surface(xi, yi, zi, cmap='RdBu_r', edgecolor='none')
ax.set_zlabel('Clusters')
ax.set_ylabel('BIC Scores')
ax.set_xlabel('SC Scores')
ax.invert_zaxis()
cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
cbar = plt.colorbar(surf, cax=cbar_ax)
cbar.ax.yaxis.tick_right()
cbar.ax.yaxis.set_label_position('right')
plt.savefig('/mnt/disk1/wyr/result_gut_net/data_driven/Train/THREED.tiff',dpi=300)
plt.show()








###########
#分界线
###########

# 代码
def kmeans(features, lowDmat):
    n_iterations = 100
    all_kmeans_labels = pd.DataFrame()
    for iteration in range(n_iterations):
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        kmeans_labels = kmeans.labels_
        all_kmeans_labels['Iteration_{}'.format(iteration + 1)] = kmeans_labels
    kmeans_labels = all_kmeans_labels.mode(axis=1).iloc[:, 0]
    kmeans_labels = np.array(kmeans_labels)
    kmeans_labels = kmeans_labels.reshape(-1, 1)
    kmeans_labels = pd.DataFrame(kmeans_labels)
    return kmeans_labels

# 基于模型聚类(精简版GMM)
def gmm(features, lowDmat):
    n_iterations = 100
    all_gmm_labels = pd.DataFrame()
    for iteration in range(n_iterations):
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(features)
        gmm_labels = gmm.predict(features)
        all_gmm_labels['Iteration_{}'.format(iteration + 1)] = gmm_labels
    # 将 gmm_labels 转换为 int32 类型
    gmm_labels = all_gmm_labels.mode(axis=1).iloc[:, 0]
    gmm_labels = np.array(gmm_labels)
    gmm_labels = gmm_labels.astype('int32')
    gmm_labels = gmm_labels.reshape(-1, 1)
    gmm_labels = pd.DataFrame(gmm_labels)
    return gmm_labels

def spectral(features, lowDmat):
    from sklearn.cluster import SpectralClustering
    n_iterations = 100
    all_spectral_labels = pd.DataFrame()
    for iteration in range(n_iterations):
        spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
        spectral_labels = spectral.fit_predict(features)
        all_spectral_labels['Iteration_{}'.format(iteration + 1)] = spectral_labels
    # 将 gmm_labels 转换为 int32 类型
    spectral_labels = all_spectral_labels.mode(axis=1).iloc[:, 0]
    spectral_labels = np.array(spectral_labels)
    spectral_labels = spectral_labels.astype('int32')
    spectral_labels = spectral_labels.reshape(-1, 1)
    spectral_labels = pd.DataFrame(spectral_labels)
    return spectral_labels


# 使用投票聚类来集成多个聚类结果
def hybrid(features, lowDmat, kmeans_labels, gmm_labels,spectral_labels):
    ensemble_labels = []
    kmeans_labels=np.array(kmeans_labels).flatten()
    gmm_labels=np.array(gmm_labels).flatten()
    spectral_labels=np.array(spectral_labels).flatten()
    for i in range(len(features)):
        votes = [kmeans_labels[i], gmm_labels[i],spectral_labels[i]]
        ensemble_label = np.argmax(np.bincount(votes))
        ensemble_labels.append(ensemble_label)
    ensemble_labels = np.array(ensemble_labels)
    ensemble_labels = ensemble_labels.reshape(-1, 1)
    ensemble_labels = pd.DataFrame(ensemble_labels)
    return ensemble_labels

kmeans_labels = kmeans(features, lowDmat)
kmeans_labels.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/k-means_4.csv', index=False, header=None)
unique_labels1 = np.unique(kmeans_labels)
color_labels = ['#87bba4','#9e3150','#fdcc7d','#e57b7f']
for label in unique_labels1:
    cluster_indices = np.where(kmeans_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results k-means')
plt.legend(['Subtype1', 'Subtype2','Subtype3','Subtype4'], loc='upper right', bbox_to_anchor=(0.3,1))
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/k-means_result_4.tiff", dpi=300)
plt.show()

gmm_labels = gmm(features, lowDmat)
gmm_labels.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/GMM_4.csv', index=False, header=None)
unique_labels2 = np.unique(gmm_labels)
color_labels = ['#87bba4',  '#9e3150','#fdcc7d','#e57b7f']
for label in unique_labels2:
    cluster_indices = np.where(gmm_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results GMM')
plt.legend(['Subtype1', 'Subtype2','Subtype3','Subtype4'], loc='upper right', bbox_to_anchor=(0.3,1))
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/GMM_result_4.tiff", dpi=300)
plt.show()

spectral_labels = spectral(features, lowDmat)
spectral_labels.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Spectral Cluster_4.csv', index=False, header=None)
unique_labels3 = np.unique(spectral_labels)
color_labels = ['#87bba4',  '#9e3150','#fdcc7d','#e57b7f']
for label in unique_labels3:
    cluster_indices = np.where(spectral_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results Spectral Clustering')
plt.legend(['Subtype1', 'Subtype2','Subtype3','Subtype4'], loc='upper right', bbox_to_anchor=(0.3,1))
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Spectral_Clustering_result_4.tiff", dpi=300)
plt.show()

ensemble_labels = hybrid(features, lowDmat,kmeans_labels, gmm_labels,spectral_labels)
ensemble_labels.to_csv('/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Final_cluster_label_4.csv', index=False, header=None)
unique_labels = np.unique(ensemble_labels)
color_labels = ['#87bba4', '#9e3150','#fdcc7d','#e57b7f']
for label in unique_labels:
    cluster_indices = np.where(ensemble_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results')
plt.legend(['Subtype1', 'Subtype2','Subtype3','Subtype4'], loc='upper right', bbox_to_anchor=(0.3,1))
plt.savefig("/mnt/disk1/wyr/result_gut_net/data_driven/Validate/Final_cluster_result_4.tiff", dpi=300)
plt.show()
#####





# ARI验证在测试集上进行预测
#####
times = 100
score = 0
kmeans_labels = np.array(kmeans_labels)
gmm_labels = np.array(gmm_labels)
spectral_labels = np.array(spectral_labels)
ensemble_labels = np.array(ensemble_labels)
ensemble_labels = ensemble_labels.flatten()
for t in range(times):
    a = np.random.randint(0, 193, size=[19, 1])
    a = a.flatten()
    test_kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(features[a])
    test_gmm = GaussianMixture(n_components=2, random_state=42).fit_predict(features[a])
    test_spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit_predict(features[a])
    test_kmeans=np.array(test_kmeans)
    test_gmm=np.array(test_gmm)
    test_spectral=np.array(test_spectral)
    test_predicted_labels = hybrid(features[a],lowDmat[a],test_kmeans, test_gmm, test_spectral)
    test_predicted_labels = np.array(test_predicted_labels)
    test_predicted_labels = test_predicted_labels.reshape(-1)
    similarity_score= adjusted_rand_score(ensemble_labels[a], test_predicted_labels)
    score = similarity_score + score
similarity_score =score / 100
print(f"模型相似度评分: {similarity_score}")# 输出相似度评分
#####





# 显著性检测
from scipy.stats import kruskal
# 添加全部标签
label_sz=pd.DataFrame(ensemble_labels)
label_nc = np.full(26,3,dtype=int)
label_nc = label_nc.reshape(-1,1)
label_nc = pd.DataFrame(label_nc)
label_all = label_nc._append(label_sz)
cluster_demographic = pd.concat([label_all.reset_index(drop=True),dataset_fc_matrix],axis=1)
# 非参数KW检验
# 全显著
from itertools import combinations
from statsmodels.stats.multitest import multipletests
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
result_df_all = pd.DataFrame(index=variable_columns_all, columns=['Kruskal-Wallis Statistic', 'P-value_all','P-value(FDR)','P-value(Bonferroni)'])
for variable in variable_columns_all:
    group_data = [cluster_demographic[cluster_demographic[category_column_all] == category][variable]
                  for category in cluster_demographic[category_column_all]]
    statistic, p_value = kruskal(*group_data)
    result_df_all.loc[variable, 'Kruskal-Wallis Statistic'] = statistic
    result_df_all.loc[variable, 'P-value_all'] = p_value
    p_values_corrected = multipletests(result_df_all['P-value_all'], alpha=0.05, method='fdr_bh')[1]
    result_df_all['P-value(FDR)'] = p_values_corrected
    p_values_corrected = multipletests(result_df_all['P-value_all'], alpha=0.05, method='Bonferroni')[1]
    result_df_all['P-value(Bonferroni)'] = p_values_corrected
print('P-Value of all', result_df_all)
result_df_all.to_csv('/mnt/disk1/wyr/result_gut_net/demographic/Validate/p_values_all_3.csv')
# 计算平均值与标准误
grouped_data = cluster_demographic.groupby(0)
means = grouped_data.mean()
std = grouped_data.std()
sem = std / np.sqrt(len(grouped_data)) # 组别个数的平方根，并非组内样本数
print('means',means)
print('std',std)
print('sem',sem)
means.to_csv('/mnt/disk1/wyr/result_gut_net/demographic/Validate/means_all_3.csv', index=False)
std.to_csv('/mnt/disk1/wyr/result_gut_net/demographic/Validate/std_all_3.csv', index=False)
sem.to_csv('/mnt/disk1/wyr/result_gut_net/demographic/Validate/sem_all_3.csv', index=False)
#####





# 在之后检测
#####
from itertools import combinations
from statsmodels.stats.multitest import multipletests
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
# 存储结果的DataFrame
result_df_all = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# 对每两组进行两两比较
for group1, group2 in combinations(cluster_demographic[category_column_all].unique(), 2):
    for variable in variable_columns_all:
        data_group1 = cluster_demographic.loc[cluster_demographic[category_column_all] == group1, variable]
        data_group2 = cluster_demographic.loc[cluster_demographic[category_column_all] == group2, variable]
        # 执行 KW 检验
        stat, p_value = kruskal(data_group1, data_group2)
        # 将结果添加到DataFrame
        row_data = [variable, group1, group2, stat, p_value]
        result_df_all = pd.concat([result_df_all, pd.DataFrame([row_data], columns=result_df_all.columns)])
p_values_corrected = multipletests(result_df_all['P-value'], alpha=0.05, method='fdr_bh')[1]
result_df_all['P-value(FDR)'] = p_values_corrected
p_values_corrected = multipletests(result_df_all['P-value'], alpha=0.05, method='Bonferroni')[1]
result_df_all['P-value(Bonferroni)'] = p_values_corrected
print(result_df_all)
result_df_all.to_csv('/mnt/disk1/wyr/result_gut_net/demographic/Validate/p_values_3.csv', index=False)


# 标签对调
original_labels = np.array([0, 1, 0, 1, 0, 1])
swapped_labels = np.where(original_labels == 0, 1, 0)

# 将原始标签中的 0 替换为 2，将 2 替换为 0
swapped_labels = np.where(original_labels == 0, 2, np.where(original_labels == 2, 0, original_labels))
spectral_labels = np.where(spectral_labels == 0, 2, np.where(spectral_labels == 2, 0, spectral_labels))
spectral_labels = pd.DataFrame(spectral_labels)