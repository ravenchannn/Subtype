#脑肠亚型分类代码-图
#Written by Raven
from __future__ import division

from imp import reload
#########################################################
#     实验目的：绘制nii的node and edge——连接+平均FC+统计+图            #
#                                        written by Raven                                         #
#########################################################
#利用这个代码计算好FC，在用matlab修改node，edge文件
#示例在D:\AAARavenResults\brain_net\差异分析结果\example\FC





#导入各种包
#####
import pandas as pd  #加载数据处理所用的包
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.stats import pearsonr, ttest_ind
import os
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
#####





# 批量复制文件名
####
# folder_path = 'G:\multisite_SZ\ROI90\cobre'
# file_names = os.listdir(folder_path)
# df = pd.DataFrame({'File Name': file_names})
# excel_file_path = 'G:\multisite_SZ\ROI90\cobre\\file_name.xlsx'
# df.to_excel(excel_file_path, index=False)
# print(f'文件名已成功复制到 {excel_file_path}')
#####





# # 计算平均FC
# #####
# df = pd.read_csv('D:\AAARavenResults\\brain_net\data_driven\label_FC.csv')
# output_directory = 'D:\\AAARavenResults\\brain_net\\label'
# os.makedirs(output_directory, exist_ok=True)
#
# # 用于存储每个标签的矩阵累加结果和计数
# label_matrix_sum = {}
# label_matrix_count = {}
#
# # 循环遍历每一行
# for index, row in df.iterrows():
#     label = row['id']  # 替换为实际的标签列名
#     file_path = row['source']  # 替换为实际的文件位置列名
#
#     # 读取txt文件并计算平均值
#     try:
#         with open(file_path, 'r') as file:
#             # 读取矩阵数据并转换为NumPy数组
#             matrix_data = np.loadtxt(file, dtype=float).reshape((90, 90))
#
#             # 累加矩阵结果
#             if label in label_matrix_sum:
#                 label_matrix_sum[label] += matrix_data
#                 label_matrix_count[label] += 1
#             else:
#                 label_matrix_sum[label] = matrix_data
#                 label_matrix_count[label] = 1
#
#             print(f"处理文件 {file_path} 完成")
#     except Exception as e:
#         print(f"处理文件 {file_path} 时发生错误: {e}")
#
# # 计算每个标签的平均矩阵并保存
# for label, matrix_sum in label_matrix_sum.items():
#     average_matrix = matrix_sum / label_matrix_count[label]
#
#     # 构建存储平均值的txt文件路径
#     output_file_path = os.path.join(output_directory, f"{label}_average_matrix.txt")
#
#     # 将平均值写入新的txt文件
#     with open(output_file_path, 'w') as output_file:
#         for row_values in average_matrix:
#             output_file.write(' '.join(map(str, row_values)) + '\n')
#
#     print(f"标签: {label}, 平均值已保存到 {output_file_path}")
# #####





# # 构成FC统计Dataframe
# import numpy as np
# import pandas as pd
# from scipy.stats import kruskal
# # 读取包含组别信息的 CSV 文件
# df = pd.read_csv('D:\AAARavenResults\\brain_net\data_driven\label_FC.csv')
# # 初始化空的矩阵列表，用于存放每个组别的矩阵
# matrix_groups = {group: [] for group in df['id'].unique()}
# # 读取每个文件中的矩阵数据并存入对应的组别
# for index, row in df.iterrows():
#     group = row['id']
#     filename = row['source']
#     with open(filename, 'r') as file:
#         matrix = file.read()
#     matrix = np.fromstring(matrix, sep=' ').reshape(90, 90)
#     matrix_flatten = matrix.reshape(1,-1)
#     matrix_groups[group].append(matrix_flatten)
# p_value_matrices = []
# A = matrix_groups[0]
# B = matrix_groups[1]
# C = matrix_groups[2]
# group0 = np.vstack(A)
# group1 = np.vstack(B)
# group2 = np.vstack(C)
# whole = np.vstack((group0,group1,group2))
# whole = pd.DataFrame(whole)
# label0 = np.full(155,0,dtype=int)
# label1 = np.full(95,1,dtype=int)
# label2 = np.full(300,2,dtype=int)
# label0 = label0.reshape(-1,1)
# label1 = label1.reshape(-1,1)
# label2 = label2.reshape(-1,1)
# label0 = pd.DataFrame(label0)
# label1 = pd.DataFrame(label1)
# label2 = pd.DataFrame(label2)
# label = label0._append(label1)
# label = label._append(label2)
# label = np.array(label)
# label = pd.DataFrame(label)
# whole.insert(0,'label',label)
# # #
# # #
# # #
# #统计FC两两之间
# from itertools import combinations
# from scipy.stats import kruskal
# from statsmodels.stats.multitest import multipletests
# category_column_all = whole.columns[0]
# variable_columns_all = whole.columns[1:]
# # 存储结果的DataFrame
# result_df = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# # 对每两组进行两两比较
# for group1, group2 in combinations(whole[category_column_all].unique(), 2):
#     for variable in variable_columns_all:
#         data_group1 = whole.loc[whole[category_column_all] == group1, variable]
#         data_group2 = whole.loc[whole[category_column_all] == group2, variable]
#         if len(set(data_group1.values.flatten())) > 1 and len(set(data_group2.values.flatten())) > 1:
#             # 进行 Kruskal-Wallis 检验
#             stat, p_value = kruskal(data_group1.values, data_group2.values)
#         else:
#             # 数值全部相同，将 p-value 设置为 1
#             stat, p_value = None, 1.0
#         # 打印结果
#         print("Kruskal-Wallis Statistic:", stat)
#         print("P-value:", p_value)
#         # 将结果添加到DataFrame
#         row_data = [variable, group1, group2, stat, p_value]
#         result_df = pd.concat([result_df, pd.DataFrame([row_data], columns=result_df.columns)])
# # 进行 FDR校正
# p_values_fdr = multipletests(result_df['P-value'], alpha=0.05, method='fdr_bh')[1]
# result_df['P-value (FDR)'] = p_values_fdr
# # 进行 Bonferroni 校正
# p_values_b = multipletests(result_df['P-value'], alpha=0.05, method='bonferroni')[1]
# result_df['P-value (Bonferroni)'] = p_values_b
# print(result_df)
# result_df.to_csv('D:\AAARavenResults\\brain_net\differ\FCFC_P_value_all.csv')
# result_df = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\brain\FC_P_value_all.csv')
# P01 = result_df.iloc[0:8100,6]
# P01 = np.array(P01)
# P01 =P01.reshape(90, 90)
# P02 = result_df.iloc[8100:16200,6]
# P02 = np.array(P02)
# P02 =P02.reshape(90, 90)
# P12 = result_df.iloc[16200:24300,6]
# P12 = np.array(P12)
# P12 =P12.reshape(90, 90)
# # 对应取最小
# P0 = P02
# P1 = P12
# # P2 = np.minimum.reduce([P12, P02])
# # 读取平均FC，找到显著的边连接
# av0 = pd.read_csv('D:\AAARavenResults\\brain_net\label\\0_average_matrix.txt', sep=' ',header=None)
# av1 = pd.read_csv('D:\AAARavenResults\\brain_net\label\\1_average_matrix.txt', sep=' ',header=None)
# av2 = pd.read_csv('D:\AAARavenResults\\brain_net\label\\2_average_matrix.txt', sep=' ',header=None)
# # P0中有值大于0.01，则将av0中对应位置的数值置0，P0不变，改av0
# A = av0-av2
# B = av1-av2
# mask0 = P0 > 0.01
# T0 = av0 < 0.5
# mask1 = P1 > 0.01
# T1 = av1 < 0.5
# A[mask0] = 0
# B[mask1] = 0
# A[T0] = 0
# B[T1] = 0
# # 输出
# A.to_csv('D:\AAARavenResults\\brain_net\differ\FC\\chord\BS1_FC.csv',index=False,header=None)
# B.to_csv('D:\AAARavenResults\\brain_net\differ\FC\\chord\BS2_FC.csv',index=False,header=None)
#
#
#
#
#
#
#
# # 对FC重排序（按照八大网络）
# import numpy as np
# import pandas as pd
# A = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\BS1_FC_av.csv',header=None)
# B = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\BS2_FC_av.csv',header=None)
# C = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\HCs_FC_av.csv',header=None)
#
# new_order = [2,	3,	14,	15,	22,	23,	30,	31,	34,	35,	64,	65,	66,	67,	84,	85,	0,	1,	16,	17,
#              18,	19,	56,	57,	68,	69,	78,	79,	80,	81,	42,	43,	44,	45,	46,	47,	48,	49,	50,	51,
#              52,	53,	54,	55,	58,	59,	60,	61,	28,	29,	32,	33,	62,	63,	6,	7,	8,	9,	10,	11,	12,	13,
#              36,	37,	40,	41,	70,	71,	72,	73,	74,	75,	76,	77,	4,	5,	20,	21,	24,	25,	26,	27,	38,
#              39,	82,	83,	86,	87,	88,	89] # 八大网络顺序
#
# new_A = A.iloc[new_order,new_order]
# new_B = B.iloc[new_order,new_order]
# # new_C = C.iloc[new_order,new_order]
#
# print(new_A)
# print(new_B)
# # print(new_C)
#
# new_A.to_csv('D:\AAARavenResults\\brain_net\differ\FC\chord\BS1_FC_reorder.csv',header=None,index=False)
# new_B.to_csv('D:\AAARavenResults\\brain_net\differ\FC\chord\BS2_FC_reorder.csv',header=None,index=False)
# new_C.to_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\HCs_FC_av_reorder.csv',header=None,index=False)




#
# # 提取八大网络的平均
# # 指定每个类别的个数
# import pandas as pd
# import numpy as np
# new_A = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\BS1_FC_av_reorder.csv', header=None).values
# new_B = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\BS2_FC_av_reorder.csv', header=None).values
# new_C = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\HCs_FC_av_reorder.csv', header=None).values
# class_sizes = [16, 14, 14, 4, 6, 8, 12, 16]
# dimension = 8
# result_matrix = np.zeros((dimension, dimension))
# row_slices = np.cumsum(class_sizes)
# col_slices = np.cumsum(class_sizes)
# for i in range(dimension):
#     start_row, end_row = row_slices[i] - class_sizes[i], row_slices[i]
#     for j in range(dimension):
#         start_col, end_col = col_slices[j] - class_sizes[j], col_slices[j]
#         sliced_matrix = new_C[start_row:end_row, start_col:end_col]
#         result_matrix[i, j] = np.mean(sliced_matrix)
# print("降维后的 8x8 矩阵:")
# print(result_matrix)
# new_A = pd.DataFrame(result_matrix)
# new_A.to_csv('D:\AAARavenResults\\brain_net\differ\FC\heat\\HCs_8X8.csv',index=False,header=None)


# DMN = new_C.iloc[0:16,0:16]
# SMN = new_C.iloc[16:30,16:30]
# VSN = new_C.iloc[30:44,30:44]
# DAN = new_C.iloc[44:48,44:48]
# VAN = new_C.iloc[48:54,48:54]
# FPN = new_C.iloc[54:62,54:62]
# SCN = new_C.iloc[62:74,62:74]
# LN = new_C.iloc[74:90,74:90]
# AV_DMN = DMN.mean().mean()
# AV_SMN = SMN.mean().mean()
# AV_VSN = VSN.mean().mean()
# AV_DAN = DAN.mean().mean()
# AV_VAN = VAN.mean().mean()
# AV_FPN = FPN.mean().mean()
# AV_SCN = SCN.mean().mean()
# AV_LN = LN.mean().mean()
# # Create a DataFrame from the average values
# mean = pd.DataFrame({
#     'DMN': [AV_DMN],
#     'SMN': [AV_SMN],
#     'VSN': [AV_VSN],
#     'DAN': [AV_DAN],
#     'VAN': [AV_VAN],
#     'FPN': [AV_FPN],
#     'SCN': [AV_SCN],
#     'LN': [AV_LN]
# })
# print("Mean DataFrame:")
# print(mean)
# mean.to_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\\Mean Network FC Healthy Controls.csv',index=False)




# # 雷达图
# import numpy as np
# import matplotlib.pyplot as plt
#
# BS1 = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Brain Subtype1.csv')
# BS2 = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Brain Subtype2.csv')
# HCs = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Healthy Controls.csv')
#
# def radar_plot(labels, values, title="Radar Plot", marker='s',color='b'):
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
#     # 将最开始的角度再添加到数组末尾，以闭合雷达图
#     values = np.concatenate((values, [values[0]]))
#     angles = np.concatenate((angles, [angles[0]]))
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True),dpi=300)
#     ax.set_theta_offset(np.pi / 2)  # 设置角度偏移
#     ax.set_theta_direction(-1)    # 设置顺时针还是逆时针，1或者-1
#     ax.plot(angles, values, color=color, linewidth=2)
#     # ax.fill(angles, values, color=color, alpha=0.5)
#     for angle, value, label in zip(angles[:-1], values[:-1], labels):
#         # 计算实心圆的坐标
#         x = angle
#         y = value
#         ax.scatter(x, y, color=color, s=50, marker='^')        # 添加实心圆
#         ax.text(x, y, f'{value:.2f}', ha='left', va='bottom')        # 添加标签
#     ax.grid(True)# 隐藏圆形网格线
#     ax.set_yticklabels([])  # 隐藏半径标签
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels,rotation=45)
#     ax.legend(loc='upper right', fontsize='small')
#     plt.title(title)
# # 示例数据
# labels = ['DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN']
# radar_plot(labels, BS1.iloc[0], title="Mean Networks FC of Brian Subtype1", marker = 'o',color='#e73847')
# radar_plot(labels, BS2.iloc[0], title="Mean Networks FC of Brian Subtype2", marker = '^',color='#a8dadb')
# radar_plot(labels, HCs.iloc[0], title="Mean Networks FC of Healthy Controls", marker = 's',color='#457b9d')
# plt.savefig('D:\AAARavenResults\\brain_net\differ\FC\\radar\Brain Network mean FC of Brain Subtype2.tiff',dpi=300)
# plt.show()





# # 雷达图
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# def radar_plot(ax, labels, values, title="Mean Network FC", color='b'):
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
#     values = np.concatenate((values, [values[0]]))
#     angles = np.concatenate((angles, [angles[0]]))
#     ax.set_theta_offset(np.pi / 2)  # 设置角度偏移
#     ax.set_theta_direction(-1)  # 设置顺时针还是逆时针，1或者-1
#     ax.plot(angles, values, label=title, color=color, linewidth=0)
#     for angle, value, label in zip(angles[:-1], values[:-1], labels):
#         x = angle
#         y = value
#         ax.scatter(x, y, color=color,marker='',alpha=0.25)
#         ax.fill(angles, values, color=color, alpha=0.25)
#         ax.text(x, y, f'{value:.3f}', ha='left', va='bottom',size=5.5)
#     ax.grid(True)
#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, rotation=45)
#     ax.legend(loc='upper right', fontsize='small')
#     ax.set_title(title)
#
#
# # 读取数据
# BS1 = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Brain Subtype1.csv')
# BS2 = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Brain Subtype2.csv')
# HCs = pd.read_csv('D:\AAARavenResults\\brain_net\differ\FC\\radar\Mean Network FC Healthy Controls.csv')
#
# # 创建一个图形和单个子图（1个极坐标）
# fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6,6), dpi=300)
# A = BS1-HCs
# B = BS2-HCs
# # 绘制三组数据
# labels = ['DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN']
# radar_plot(ax, labels, A.iloc[0], title='BS1',color='#e73847')
# radar_plot(ax, labels, B.iloc[0], title='BS2',color='#a8dadb')
#
# # 调整子图布局
# fig.tight_layout()
#
# # 保存图像
# plt.savefig('D:\AAARavenResults\\brain_net\differ\FC\\radar\Changes FC1.tiff', dpi=300)
# plt.show()







# draw chord plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
    def evaluate(self, t):
        n = len(self.control_points) - 1
        return np.sum([self.control_points[i] * self.bernstein_poly(i, n, t) for i in range(n + 1)], axis=0)
    def bernstein_poly(self, i, n, t):
        return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
def draw_chord_plot(df):
    plt.rcParams.update({'font.size': 30})
    class_sizes = [16, 14, 14, 4, 6, 8, 12, 16]
    class_names = ['DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN']
    custom_colors = ['#008F7A','#0089BA', '#2C73D2', '#845EC2', '#D65DB1', '#FF6F91', '#FF9671', '#FFC75F']
    custom_cmap = ListedColormap(custom_colors)
    cmap = custom_cmap
    # cmap = plt.get_cmap('magma', len(class_sizes))
    colors = [cmap(i) for i in range(len(class_sizes))]
    plt.figure(figsize=(20, 16))
    plt.pie(class_sizes, labels=class_names, startangle=0, colors=colors, wedgeprops=dict(width=0.5, linewidth=15, edgecolor='white'), textprops={'fontsize': 30})
    centre_circle = plt.Circle((0, 0),0.9, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    sorted_values = df.values.flatten()
    sorted_values.sort()
    top_values = sorted_values[-22:][::-1]  # 取前 10 大的值
    down_values = sorted_values[:22][::-1]  # 取前 10 大的值
    # 绘制连接曲线
    for i in range(90):
        for j in range(i + 1, 90):
            value = df.iloc[i, j]
            if value in top_values:
                angle_i = i * 360 / 90
                angle_j = j * 360 / 90
                x_i, y_i = 0.87 * np.cos(np.radians(angle_i)), 0.87 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.87 * np.cos(np.radians(angle_j)), 0.87 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#D86161'##6D9AEF
                # 线条粗细和颜色
                max_line_width = 10
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1, 45)))
                # 添加曲线
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    for i in range(90):# 减小连接
        for j in range(i + 1, 90):
            value = df.iloc[i, j]
            if value in down_values:
                angle_i = i * 360 / 90
                angle_j = j * 360 / 90
                x_i, y_i = 0.87 * np.cos(np.radians(angle_i)), 0.87 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.87 * np.cos(np.radians(angle_j)), 0.87 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#6D9AEF'
                max_line_width = 10
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1, 45)))
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    # 外周圆环图例
    class_names = ['DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN']
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=30, label=l)
                      for c, l in zip(colors, class_names)]
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.95, 0.8), loc='upper left',frameon=False)
    plt.axis('equal')
# 使用示例
df = pd.read_csv('D:\AAARavenResults\\brain_net_whole\differ\FC\chord\BS1_FC_reorder.csv',header=None)
draw_chord_plot(df)
plt.savefig('D:\AAARavenResults\\brain_net_whole\differ\FC\chord\BS1_FC.tiff',dpi=300)
plt.show()





# #计算个体连接8*8FC
# import pandas as pd
# import numpy as np
# df = pd.read_csv('D:\AAARavenResults\\brain_net\data_driven\label_FC.csv')
# # 用于存储每次循环生成的1x64矩阵
# individual_matrices = []
# class_sizes = [16, 14, 14, 4, 6, 8, 12, 16]
# dimension = 8
# row_slices = np.cumsum(class_sizes)
# col_slices = np.cumsum(class_sizes)
# # 循环遍历每个人的数据
# for index, row in df.iterrows():
#     file_path = row['source']
#     result_matrix = np.zeros((dimension, dimension))
#     with open(file_path, 'r') as file:
#         matrix_data = np.loadtxt(file, dtype=float).reshape((90, 90))
#     for i in range(dimension):
#         start_row, end_row = row_slices[i] - class_sizes[i], row_slices[i]
#         for j in range(dimension):
#             start_col, end_col = col_slices[j] - class_sizes[j], col_slices[j]
#             sliced_matrix = matrix_data[start_row:end_row, start_col:end_col]
#             result_matrix[i, j] = np.mean(sliced_matrix)
#     # 将 result_matrix 转换为一维数组 A，并添加到列表
#     A = result_matrix.reshape(1, -1)
#     individual_matrices.append(A)
# # 使用 np.vstack 将列表中的数组堆叠成一个大数组
# stacked_matrix = np.vstack(individual_matrices)
# # 创建行名和列名
# row_names = df.iloc[:, 0].values  # 使用第一列作为行名
# column_names = [
#     'DMN_DMN', 'SMN_DMN', 'VSN_DMN', 'DAN_DMN', 'VAN_DMN', 'FPN_DMN', 'SCN_DMN', 'LN_DMN',
#     'DMN_SMN', 'SMN_SMN', 'VSN_SMN', 'DAN_SMN', 'VAN_SMN', 'FPN_SMN', 'SCN_SMN', 'LN_SMN',
#     'DMN_VSN', 'SMN_VSN', 'VSN_VSN', 'DAN_VSN', 'VAN_VSN', 'FPN_VSN', 'SCN_VSN', 'LN_VSN',
#     'DMN_DAN', 'SMN_DAN', 'VSN_DAN', 'DAN_DAN', 'VAN_DAN', 'FPN_DAN', 'SCN_DAN', 'LN_DAN',
#     'DMN_VAN', 'SMN_VAN', 'VSN_VAN', 'DAN_VAN', 'VAN_VAN', 'FPN_VAN', 'SCN_VAN', 'LN_VAN',
#     'DMN_FPN', 'SMN_FPN', 'VSN_FPN', 'DAN_FPN', 'VAN_FPN', 'FPN_FPN', 'SCN_FPN', 'LN_FPN',
#     'DMN_SCN', 'SMN_SCN', 'VSN_SCN', 'DAN_SCN', 'VAN_SCN', 'FPN_SCN', 'SCN_SCN', 'LN_SCN',
#     'DMN_LN', 'SMN_LN', 'VSN_LN', 'DAN_LN', 'VAN_LN', 'FPN_LN', 'SCN_LN', 'LN_LN'
# ]
# # 将 NumPy 数组转换为 DataFrame，指定行名和列名
# stacked_matrix_df = pd.DataFrame(stacked_matrix, index=row_names, columns=column_names)
# # 打印结果
# print("带有行名和列名的 550x64 矩阵:")
# print(stacked_matrix_df)
# stacked_matrix_df.to_csv('D:\AAARavenResults\\brain_net\label\FC.csv')
# #####





import numpy as np
import pandas as pd
df = pd.read_csv('D:\AAARavenResults\\brain_net_whole\differ\FC\chord\BS2_FC.csv',header=None)
sorted_values = np.sort(df.values.flatten())
down_values = sorted_values[:22][::-1]
processed_values = np.zeros_like(df.values)
processed_values[np.isin(df.values, down_values)] = df.values[np.isin(df.values, down_values)]
positive_values = sorted_values[sorted_values > 0]
processed_values[np.isin(df.values, positive_values)] = df.values[np.isin(df.values, positive_values)]
processed_df = pd.DataFrame(processed_values)
print(processed_df)
processed_df.to_csv('D:\AAARavenResults\\brain_net_whole\differ\FC\\brain\BS2_FC.csv',header=None)