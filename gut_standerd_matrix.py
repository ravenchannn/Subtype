#脑肠亚型分类代码-肠
#Written by Raven
from __future__ import division
#########################################################
#                                 实验目的：个体肠网络聚类标签                                 #
#                                        written by Raven                                         #
#########################################################





import pandas as pd
import numpy as np
#2类
# 读取CSV文件
label = pd.read_csv("/mnt/disk1/wyr/result_gut_net/data_driven/Train/Final_cluster_label_3.csv", encoding='gb18030', header=None)  # 替换成您想要的标签
# 添加全部标签
label_nc = np.full(97,3,dtype=int) #97/26
label_nc = label_nc.reshape(-1,1)
label_nc = pd.DataFrame(label_nc)
label_all = label_nc._append(label)
label_all = np.array(label_all)
label_all = pd.DataFrame(label_all)
# label_all.to_csv('/mnt/disk1/wyr/result_brain_net/all_final_label.csv',header=None,index=False)










# 读取要加标签的文件
input_file = "/mnt/disk1/wyr_data/SZ_gut316/Train/regenus.csv"  # 替换成您的输入文件名
output_file = '/mnt/disk1/wyr_data/datasets_SZ_gut_net_label/Train/hhhhh/regenus.csv'  # 替换成您的输出文件名
# 读取CSV文件到DataFrame
fc_matrix = pd.read_csv(input_file)
# 在DataFrame的最前列添加标签列
fc_matrix.insert(0, 'id', label_all)
# 将修改后的DataFrame保存为新的CSV文件
fc_matrix.to_csv(output_file, index=False)
print('ok')