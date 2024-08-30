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
#                          实验目的：个体脑-肠网络数据驱动模型2                           #
#                                        written by Raven                                         #
#########################################################
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####

random_seed = 42
np.random.seed(42)
np.random.seed(random_seed)
Validation=pd.read_csv("/mnt/disk1/wyr_data/bg_net_sz/Validation/set.csv")
data = Validation[12:]
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('患者', data) #取所需要的数据





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
plt.savefig('/mnt/disk1/wyr/bg_sz/data_driven/Validation/ICA.tiff', dpi=300)
plt.show()
scaler = StandardScaler()
lowDmat = scaler.fit_transform(lowDmat)
print('降维后',lowDmat)
print('重构后',reconMat_1)
lowDmat = pd.DataFrame(lowDmat)
lowDmat.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Validation/ICA.csv',index=False,header=None)
lowDmat = np.array(lowDmat)
#####





#自编码器数据增强
#####
from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense
from keras.optimizers import Adam
# 构建深层聚类模型
input_dim = reconMat_1.shape[1]  # 输入维度
latent_dim = 1  # 潜在空间维度
# 定义函数来创建和训练深度聚类模型
def Validation_deep_clustering_model(input_data):
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
    noisy = data + 0.1 * np.random.randn(*reconMat_1.shape)  # 0.1是噪声水平
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
    features = Validation_deep_clustering_model(data)
    all_features.append(features)
features = np.mean(all_features, axis=0)
scaler = StandardScaler()
features = scaler.fit_transform(features)
features_df=pd.DataFrame(features)
features_df.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Validation/Autoencoder_results.csv', index=False, header=None)
#####





# 混合聚类
#####
# 导入各种包
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
# SC系数
clusters = [2,3,4,5,6,7,8,9]
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
sc_df.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Validation/SC.csv', index=False)
plt.plot(clusters, sc_scores, color='#bf1e2e',marker='o')
plt.title('Silhouette Coefficient')
plt.xticks(range(2, 10, 1))
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Components_Silhouette Coefficient')
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Validation/SC.tiff",dpi=300)
plt.show()

# BIC计算
bic_gmm = []
for n_components in range(2, 10):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)
    bic_gmm.append(gmm.bic(features))
    print('K = %s, bic= %0.03f' % (n_components, gmm.bic(features)))
bic_df = pd.DataFrame(list(zip(clusters, bic_gmm)), columns=['Number of Clusters', 'BIC'])
bic_df.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Validation/bic.csv', index=False)
plt.plot(clusters, bic_gmm, color='#1d3557',marker='o')
plt.title('Cluster Components_BIC')
plt.xticks(range(2, 10, 1))
plt.xlabel('Number of Clusters')
plt.ylabel('Cluster Components_BIC')
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Validation/bic.tiff",dpi=300)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import adjusted_rand_score
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.mixture import GaussianMixture
# from scipy.interpolate import griddata
# # 聚类簇数范围
# features=pd.read_csv("/mnt/disk1/wyr/bg_sz/data_driven/Validation/Autoencoder_results.csv",header=None)
# clusters = [2, 3, 4, 5, 6, 7, 8, 9]
# subplot_counter = 1
# sc_scores = []
# for t in clusters:
#     subplot_counter += 1
#     kmeans_model = KMeans(n_clusters=t).fit(features)
#     # 绘制轮廓系数与不同类簇数量的直观显示图
#     sc_score = silhouette_score(features, kmeans_model.labels_, metric='euclidean')
#     sc_scores.append(sc_score)
#     print('K = %s, Silhouette Coefficient= %0.03f' %(t, sc_score))
# bic_gmm = []
# for n_components in range(2,10):
#     gmm = GaussianMixture(n_components=n_components, random_state=42)
#     gmm.fit(features)
#     bic_gmm.append(gmm.bic(features))
#     print('K = %s, bic= %0.03f' % (n_components, gmm.bic(features)))
# z =sc_scores
# y = bic_gmm
# x = clusters
# xi = np.linspace(min(x), max(x), 1000)
# yi = np.linspace(min(y), max(y), 1000)
# xi, yi = np.meshgrid(xi, yi)
# zi = griddata((x, y), z, (xi, yi), method='cubic')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(xi, yi, zi, cmap='RdBu_r', edgecolor='none')
# ax.plot_surface(xi, yi, zi, cmap='RdBu_r', edgecolor='none')
# ax.set_zlabel('SC Scores')
# ax.set_ylabel('BIC Scores')
# ax.set_xlabel('Clusters')
# ax.invert_zaxis()
# cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
# cbar = plt.colorbar(surf, cax=cbar_ax)
# cbar.ax.yaxis.tick_right()
# cbar.ax.yaxis.set_label_position('right')
# plt.savefig('/mnt/disk1/wyr/bg_sz/data_driven/Validation/THREED.tiff',dpi=300)
# plt.show()





# 混合聚类：
features=pd.read_csv("/mnt/disk1/wyr/bg_sz/data_driven/Train/Autoencoder_results.csv",header=None)
lowDmat=pd.read_csv("/mnt/disk1/wyr/bg_sz/data_driven/Train/ICA.csv",header=None)
features=np.array(features)
lowDmat=np.array(lowDmat)
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
kmeans_labels.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Train/k-means.csv', index=False, header=None)
unique_labels1 = np.unique(kmeans_labels)
color_labels = ['#b1c44d','#edb0af']
for label in unique_labels1:
    cluster_indices = np.where(kmeans_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results k-means')
plt.legend(['Subtype1', 'Subtype2'], loc='upper right', bbox_to_anchor=(0.3, 1))
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Train/k-means_result.tiff", dpi=300)
plt.show()

gmm_labels = gmm(features, lowDmat)
gmm_labels.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Train/GMM.csv', index=False, header=None)
unique_labels2 = np.unique(gmm_labels)
color_labels = ['#b1c44d','#edb0af']
for label in unique_labels2:
    cluster_indices = np.where(gmm_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results GMM')
plt.legend(['Subtype1', 'Subtype2'], loc='upper right', bbox_to_anchor=(0.3, 1))
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Train/GMM_result.tiff", dpi=300)
plt.show()

spectral_labels = spectral(features, lowDmat)
spectral_labels.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Train/Spectral Cluster.csv', index=False, header=None)
unique_labels3 = np.unique(spectral_labels)
color_labels = ['#b1c44d','#edb0af']
for label in unique_labels3:
    cluster_indices = np.where(spectral_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results Spectral Clustering')
plt.legend(['Subtype1', 'Subtype2'], loc='upper right', bbox_to_anchor=(0.3, 1))
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Train/Spectral_Clustering_result.tiff", dpi=300)
plt.show()

ensemble_labels = hybrid(features, lowDmat,kmeans_labels, gmm_labels,spectral_labels)
ensemble_labels.to_csv('/mnt/disk1/wyr/bg_sz/data_driven/Train/Final_cluster_label.csv', index=False, header=None)
unique_labels = np.unique(ensemble_labels)
color_labels = ['#b1c44d','#edb0af']
for label in unique_labels:
    cluster_indices = np.where(ensemble_labels == label)[0]
    plt.scatter(lowDmat[cluster_indices, 0], lowDmat[cluster_indices, 1], color=color_labels[label], alpha=0.8,
                label='Cluster {}'.format(label))
plt.title('Clustering Results')
plt.legend(['Subtype1', 'Subtype2'], loc='upper right', bbox_to_anchor=(0.3, 1))
plt.savefig("/mnt/disk1/wyr/bg_sz/data_driven/Train/Final_cluster_result.tiff", dpi=300)
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
features = np.array(features)
for t in range(times):
    a = np.random.randint(0, 33, size=[10, 1])
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
