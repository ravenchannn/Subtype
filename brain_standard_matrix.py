#脑肠亚型分类代码-脑
#Written by Raven
from __future__ import division
#########################################################
#                                 实验目的：个体脑网络聚类标签                                 #
#                                        written by Raven                                         #
#########################################################




import pandas as pd
import numpy as np

# 读取CSV文件
label = pd.read_csv("/mnt/disk1/wyr/result_brain_net/data_driven/Validate/Final_cluster_label.csv", encoding='gb18030', header=None)  # 替换成您想要的标签
# 添加国基（血 + 肠）标签
# label_98_sz  = label[140:183] # 左闭右开
# label_98_sz = label
# label_98_nc = np.full(55,2,dtype=int)
# label_98_nc = label_98_nc.reshape(-1,1)
# label_98_nc = pd.DataFrame(label_98_nc)
# label_98 = label_98_nc._append(label_98_sz)
# label_98 = np.array(label_98)
# label_98 = pd.DataFrame(label_98)
# label_98.to_csv('/mnt/disk1/wyr/result_gut_net/final_label.csv',header=None,index=False)

# 添加全部标签
label_nc = np.full(232,2,dtype=int)
label_nc = label_nc.reshape(-1,1)
label_nc = pd.DataFrame(label_nc)
label_all = label_nc._append(label)
label_all = np.array(label_all)
label_all = pd.DataFrame(label_all)
# label_all.to_csv('/mnt/disk1/wyr/result_brain_net/all_final_label.csv',header=None,index=False)










# 读取要加标签的文件
input_file = '/mnt/disk1/wyr_data/SZ_brain550/GLM_removed_covariate/ALFF.csv'  # 替换成您的输入文件名
output_file = '/mnt/disk1/wyr_data/datasets_SZ_brain_net_label/Validate/ALFF.csv'  # 替换成您的输出文件名

# 读取CSV文件到DataFrame
fc_matrix = pd.read_csv(input_file)

# 在DataFrame的最前列添加标签列
K= fc_matrix.iloc[0:232]  #232:300
J = fc_matrix.iloc[300:483]   #483:
stacked_data = pd.concat([K, J], axis=0)
fc_matrix = stacked_data.reset_index(drop=True)
fc_matrix.insert(0, 'id', label_all)

# 将修改后的DataFrame保存为新的CSV文件
fc_matrix.to_csv(output_file, index=False)
print('ok')