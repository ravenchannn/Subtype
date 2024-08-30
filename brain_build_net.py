#脑肠亚型分类代码-脑
#Written by Raven
from __future__ import division

import math

#########################################################
#                               实验目的：绘制个体脑网络热图                                   #
#                                        written by Raven                                         #
#########################################################






#导入各种包
#####
import matplotlib.pyplot as plt  #加载画图的包
import pandas as pd
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####




# # 读取个体脑网络
# folder_path = 'D:\AAARavenResults\\brain_net\label'
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
#
# for file_name in file_names:
#     file_path = os.path.join(folder_path, file_name)
#     data = np.loadtxt(file_path)
#     sns.heatmap(data,cmap='plasma')
#     plt.xlabel(f'Brain Correlation Matrix of {file_name}')
#     # plt.xlabel(f'Brain Correlation Matrix of {file_name}')
#     plt.savefig(f'D:\AAARavenResults\\brain_net\label {file_name} .tiff',dpi=300)
#     # plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/Mean Brain HeatMap of Subtype1')
#     plt.show()




# # 群体脑网络如何绘制
# data = pd.read_csv("/mnt/disk1/wyr/result_brain_net/FC/heat/HCs_FC_av.csv",header=None)
# sns.heatmap(data,cmap='magma')
# plt.xlabel('Average Brain HeatMap of Healthy Controls')
# xticks_position = np.arange(0, 100, 10)
# xticks_labels = [str(i) for i in xticks_position]  # 将刻度位置转换为字符串
# plt.xticks(xticks_position, xticks_labels,rotation=0)
# plt.yticks(xticks_position, xticks_labels)
# plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/heat/Mean Brain HeatMap of Healthy Control.tiff',dpi=300)
# plt.show()
# print('done')




# 重排序热图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("/mnt/disk1/wyr/result_brain_net/FC/heat/BS1_FC_av_reorder.csv",header=None)

# 设定行列的标签，这里以前5行为系列A，后4行为系列B
row_labels = ['DMN'] * 16 + ['SMN'] * 14 + ['VSN'] * 14 + ['DAN'] * 4 + ['VAN'] * 6 + ['FPN'] * 8 + ['SCN'] * 12 + ['LN'] * 16
col_labels = row_labels

# 创建一个 DataFrame 用于绘制黑线
line_data = pd.DataFrame(index=data.index)  # 指定 index

# 添加黑线标记列
line_data['Line'] = ['DMN'] * 16 + ['SMN'] * 14 + ['VSN'] * 14 + ['DAN'] * 4 + ['VAN'] * 6 + ['FPN'] * 8 + ['SCN'] * 12 + ['LN'] * 16

# 绘制热图
plt.figure(figsize=(10, 8))  # 设置图的大小
sns.heatmap(data, cmap='magma',annot=False)

# 绘制标签之间的黑线
current_label = None
for i in range(len(line_data)):
    label = line_data['Line'][i]
    if label != current_label:
        plt.axhline(i, color='white', lw=2)
        current_label = label

        # 在列标签之间也画黑线
        plt.axvline(i, color='white', lw=2)

rect1 = Rectangle((0, 0), len(col_labels), len(row_labels) + 1, linewidth=2, edgecolor='white', facecolor='none')
rect2 = Rectangle((len(col_labels), 0), 1, len(row_labels) + 1, linewidth=2, edgecolor='white', facecolor='none')
rect3 = Rectangle((0, len(row_labels)), len(col_labels) + 1, 1, linewidth=2, edgecolor='white', facecolor='none')

plt.gca().add_patch(rect1)
plt.gca().add_patch(rect2)
plt.gca().add_patch(rect3)

# 添加标签
plt.title('Brian Network FC of Brain Subtype1')
plt.xticks([8,23,37,45,52,58,68,82],[ 'DMN','SMN','VSN','DAN','VAN','FPN','SCN','LN'],rotation=0,fontsize=15)
plt.yticks([8,23,37,46,51,58,68,82],[ 'DMN','SMN','VSN','DAN','VAN','FPN','SCN','LN'],rotation=0,fontsize=15)

# 保存图像
plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/heat/Brian Network FC of BS1 all.tiff',dpi=300)

# 显示图像
plt.show()

print('done')





# 平均热图
# 提取八大网络的平均
# 指定每个类别的个数
import pandas as pd
import numpy as np
BS1 = pd.read_csv("/mnt/disk1/wyr/result_brain_net/FC/heat/BS1_8X8.csv", header=None).values
BS2 = pd.read_csv("/mnt/disk1/wyr/result_brain_net/FC/heat/BS2_8X8.csv", header=None).values
HCs = pd.read_csv("/mnt/disk1/wyr/result_brain_net/FC/heat/HCs_8X8.csv", header=None).values
# 设定行列的标签，这里以前5行为系列A，后4行为系列B
row_labels = ['DMN','SMN','VSN','DAN','VAN','FPN','SCN','LN']
col_labels = row_labels
plt.figure(figsize=(10, 8))  # 设置图的大小
sns.heatmap(HCs, cmap='magma',annot=False)
for i in range(len(row_labels) + 1):
    plt.axhline(i, color='white', linewidth=1)
    plt.axvline(i, color='white', linewidth=1)
plt.title('Brian Network FC of Healthy Controls')
plt.xticks(np.arange(len(row_labels))+0.5,row_labels,rotation=0,fontsize=15)
plt.yticks(np.arange(len(row_labels))+0.5,row_labels,rotation=0,fontsize=15)
plt.tick_params(axis='both', which='both', pad=10)
# 保存图像
plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/heat/Brian Network FC of HCs.tiff',dpi=300)
# 显示图像
plt.show()
print('done')