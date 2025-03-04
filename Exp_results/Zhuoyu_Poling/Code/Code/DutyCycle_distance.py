import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Qtagg')
import matplotlib.pyplot as plt

import os
from scipy import*
from scipy import signal
from scipy.signal import find_peaks
row = np.arange(0,18,1)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
g = globals()
import matplotlib.colors as mcolors
from scipy.optimize import *
from itertools import *
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from numpy import sqrt, min,max,sin,cos
# from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
from scipy.signal import convolve, find_peaks
from scipy.signal.windows import gaussian
from scipy.fftpack import fft, fftfreq


# 定义数据文件的相对路径
data_file = Path("..") / ".."/"20250225" / "Confocal" / "20250225-2035-40_confocal_xy_data.dat"
# 确保路径正确解析
print(f"解析后的绝对路径: {data_file.resolve()}")

# 检查文件是否存在
if data_file.exists():
    print("文件存在，准备读取数据")
 # 读取文件
    data = pd.read_csv(data_file, delimiter="\t", skiprows=row)
    print(data.head())
else:
    print("error 文件不存在，请检查路径是否正确！")



# removing unwanted columns
data = data.drop(["count rate /Dev2/AI1 (Hz)","count rate /Dev2/AI0 (Hz)",'z position (m)',"Unnamed: 6"], axis='columns')
#visualise the data

# sort the data according to the prefence
# must be sorted for depth analysis
data_sort = data.sort_values(by=['#x position (m)'],ignore_index = True)

# Shifting the origin to (0,0) and rescale the coordinate to real distance
data['#x position (m)'] = (data['#x position (m)']-data['#x position (m)'].min())*1e6*0.606
data['y position (m)'] = ((data['y position (m)']-data['y position (m)'].min())*1e6)*0.368

# 复制数据并筛选 poling 区域
df = data.copy()
df.drop(df[df['#x position (m)'] < 30].index, inplace=True)
#df.drop(df[df['#x position (m)'] >20].index, inplace=True)
#df.drop(df[df['y position (m)'] > 15].index, inplace=True)
#df.drop(df[df['y position (m)'] < 5].index, inplace=True)

# 提取 X 位置 和 SHG 信号
x_data = df['#x position (m)'].values
shg_intensity = df['count rate /Dev2/Ctr3 (Hz)'].values  # SHG 计数

# **Step 1: 归一化 SHG 信号**
shg_intensity = np.sqrt(shg_intensity)  # SHG 强度 ∝ \( d_{33}^2 \)，取平方根
shg_intensity /= np.max(shg_intensity)  # 归一化到 [0,1]



df['shg_intensity'] = shg_intensity  # 把平滑后的数据加入 DataFrame 里

# **自动生成文件名前缀**
file_prefix = data_file.stem  # 获取输入数据文件名
save_dir = Path("..") / ".." / "Processed_Figures"
save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
y_e = np.linspace(round(5*(len(df))/35),round(20*(len(df))/35),num=40)
# **Step 1: 设定 y_e 的扫描范围**
y_e_values = df["y position (m)"].iloc[y_e]
duty_cycles = []  # 存储计算出的占空比
valid_y_e = []  # 存储有效的 y_e
width_std = []

for y_e in y_e_values:
    # **Step 2: 提取 X 方向数据**
    X_e, Z_e = [], []
    for i in range(len(df)):
        y = df["y position (m)"].iloc[i]
        if np.isclose(y, y_e, atol=0.1):  # 找到接近 y_e 的数据点
            x = df["#x position (m)"].iloc[i]
            z = df['shg_intensity'].iloc[i]
            X_e.append(x)
            Z_e.append(z)

    x_e = np.asarray(X_e)
    z_e = np.asarray(Z_e)

    # **Step 3: 确保数据非空**
    if len(x_e) == 0 or len(z_e) == 0:
        continue  # 跳过该 y_e

    # **Step 4: 高斯滤波去噪**
    gauss_window = gaussian(4, std=3)
    gauss_window /= gauss_window.sum()
    z_e_smooth = convolve(z_e, gauss_window, mode='same')

    # **Step 5: 设定最小间距 distance**
    min_peak_distance = 1 / (x_e[1] - x_e[0])  # 计算对应的数据点数
    dark_peaks, _ = find_peaks(-z_e_smooth, distance=min_peak_distance)  # 找极小值

    # **Step 6: 计算相邻 peak 之间的距离**
    if len(dark_peaks) > 1:
        all_peak_distances = np.diff(x_e[dark_peaks])  # 计算所有相邻极小值之间的距离

        # **Step 7: 只保留 1.4—2 μm 之间的间距**
        filtered_peak_distances = [d for d in all_peak_distances if 1.2 <= d <= 2.2]
        width_std_cup = np.std(filtered_peak_distances)

        if len(filtered_peak_distances) > 0:
            avg_distance = np.mean(filtered_peak_distances)  # 计算筛选后的平均间距
        else:
            avg_distance = None  # 如果筛选后没有数据，则设为 None
    else:
        avg_distance = None

    # **Step 8: 计算占空比**
    if avg_distance is not None and len(filtered_peak_distances) > 0:
        duty_cycle = np.median(filtered_peak_distances) / (2 * avg_distance)
        width_std.append(width_std_cup)
        # **修正：添加数据到列表**
        duty_cycles.append(duty_cycle)  # 存储计算出的占空比
        valid_y_e.append(y_e)  # 存储有效的 y_e
        print(f"Duty Cycle: {duty_cycle:.2f}")
    else:
        print("Error: Cannot compute duty cycle (no valid peak distances).")

# 绘制 poling duty cycle 和标准差分布
fig, ax1 = plt.subplots(figsize=(10, 6))
# Duty Cycle 曲线
color1 = 'tab:blue'
ax1.set_xlabel("Y Position (μm)", fontsize=14)
ax1.set_ylabel("Duty Cycle", color=color1, fontsize=14)
ax1.plot(valid_y_e, duty_cycles, marker='o', linestyle='-', color=color1, label="Duty Cycle")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.1)  # 限制 duty cycle 在 0 到 1 之间



# 添加次坐标轴表示标准差
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Standard Deviation of Peak Distance", color=color2, fontsize=14)
ax2.plot(valid_y_e, width_std, marker='s', linestyle='--', color=color2, label="STD Width")
ax2.tick_params(axis='y', labelcolor=color2)

# 标题 & 图例
fig.suptitle("Poling Duty Cycle and Standard Deviation Distribution", fontsize=16)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

# 保存图片
plot_filename = f"{file_prefix}_DutyCycle_vs_Y.png"
plt.savefig(save_dir / plot_filename, dpi=600, transparent=True, bbox_inches='tight')
plt.show()

print(f"📊 Plot saved as {plot_filename}")