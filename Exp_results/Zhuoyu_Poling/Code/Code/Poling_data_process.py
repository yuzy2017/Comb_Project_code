import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.signal.windows import gaussian
from scipy.fftpack import fft, fftfreq
from pathlib import Path
import glob
import scipy.interpolate

row = np.arange(0, 18, 1)

# **Step 1: 设置实验数据目录**
date = "20250304"  # 需要处理的日期
data_dir = Path("..") / ".." / date / "Confocal"  # 数据目录
save_dir = Path("..") / ".." / "Processed_Figures" / date  # 结果保存路径
save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

# **创建日志文件**
log_file_path = save_dir / "processing_log.txt"
with open(log_file_path, "w", encoding="utf-8") as log_file:
    def log_print(*args, **kwargs):
        """同时输出到终端 & 写入日志"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)

    log_print(f"📂 处理日期: {date}")

    # **Step 2: 查找所有符合 _confocal_xy_data.dat 结尾的文件**
    data_files = sorted(glob.glob(str(data_dir / "*_confocal_xy_data.dat")))
    if not data_files:
        log_print(f"❌ 没有找到 {date} 目录下符合 *_confocal_xy_data.dat 的文件，请检查路径！")
    else:
        log_print(f"📂 发现 {len(data_files)} 个数据文件，开始处理...")

    # **Step 3: 遍历所有文件**
    for data_file in data_files:
        data_file = Path(data_file)
        log_print(f"📊 处理文件: {data_file.name}")

        # **Step 3.1: 读取数据**
        try:
            data = pd.read_csv(data_file, delimiter="\t", skiprows=row)
        except Exception as e:
            log_print(f"❌ 读取失败: {data_file.name}, 错误: {e}")
            continue

        # **Shifting the origin & 清理数据**
        data['#x position (m)'] = (data['#x position (m)'] - data['#x position (m)'].min()) * 1e6 * 0.606
        data['y position (m)'] = ((data['y position (m)'] - data['y position (m)'].min()) * 1e6) * 0.368
        #for rings
        data.drop((data[data['#x position (m)'] < 20].index), inplace=True)
        data.drop((data[data['#x position (m)'] >50].index), inplace=True)
        #data.drop((data[data['#x position (m)'] < 30].index), inplace=True)

        # 提取 X、Y、Z 数据
        x = data['#x position (m)'].values
        y = data['y position (m)'].values
        z = data['count rate /Dev2/Ctr3 (Hz)'].values
        z = np.sqrt(z) / np.sqrt(np.max(z))
        data['shg_intensity'] =z

        # 计算 98% 最大值，避免极端点影响
        v_max = np.percentile(z, 98)

        # **Step 3.2: 生成并保存 SHG 图像**
        grid_x, grid_y = np.linspace(x.min(), x.max(), 800), np.linspace(y.min(), y.max(), 800)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_z = scipy.interpolate.griddata((x, y), z, (grid_x, grid_y), method='cubic')

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        im = ax.imshow(grid_z * 2, extent=[x.min(), x.max(), y.min(), y.max() * 0.9], origin='lower',
                       cmap='binary', aspect='auto', vmin=0, vmax=v_max)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHG Intensity (counts/sec)', fontsize=12)
        ax.set_xlabel('X position (μm)', fontsize=12)
        ax.set_ylabel('Y position (μm)', fontsize=12)
        plt.savefig(save_dir / f"{data_file.stem}_SHG.png", dpi=600, transparent=True, bbox_inches='tight')
        plt.close(fig)
        # define the line for electrode refe
        #
        # rence
        y_e = round(15 * (len(data)) / 35)
        # **Step 1: 提取 X 方向数据**
        X_e, Z_e = [], []
        if y_e >= len(data):
            y_e = len(data) - 1  # 避免索引越界
        y_ref = data["y position (m)"].iloc[y_e]

        for i in range(len(data)):
            y = data["y position (m)"].iloc[i]
            if np.isclose(y, y_ref, atol=1e-10):
                x = data["#x position (m)"].iloc[i]
                z = data['shg_intensity'].iloc[i]  # 归一化后的 d33 值
                X_e.append(x)
                Z_e.append(z)

        x_e = np.asarray(X_e)
        z_e = np.asarray(Z_e)
        vmin = data['shg_intensity'].min()
        # vmax = df['count rate /Dev2/Ctr3 (Hz)'].max()
        vmax = 0.98

        fig=plt.figure(figsize=(6,9))
        # plot the data
        plt.scatter(x=data['#x position (m)'], y=data['y position (m)']
                    , c=data['shg_intensity'], cmap='magma', vmin=vmin, vmax=vmax)
        plt.axhline(y=data['y position (m)'].iloc[y_e], color='black', linestyle='-.', linewidth=5)
        plt.axhline(y=data['y position (m)'].iloc[round(4 * (len(data)) / 16)], color='r', linestyle='-.', linewidth=5)
        # plt.ylim(0,22)
        # plt.axis('off')
        plt.xlabel('X position ($\mu$m)', size=24)
        plt.ylabel('Y position ($\mu$m)', size=24)
        plt.tick_params(axis='both', labelsize=24)
        cbar = plt.colorbar()
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=20)
        # cbar.ax.set_title('SHG intensity (counts/sec)',fontsize=20,rotation = 90)
        plt.tick_params(axis='both', labelsize=24)
        file_prefix = data_file.stem  # 提取数据文件名
        fig.savefig(save_dir / f"{file_prefix}_Sample_position.png", dpi=600, transparent=True, bbox_inches='tight')
        plt.close(fig)





        # 确保数据非空
        if len(x_e) == 0 or len(z_e) == 0:
            raise ValueError("Error: No valid data points found!")
        gauss_window = gaussian(4, std=3)
        gauss_window /= gauss_window.sum()
        z_e_smooth = convolve(z_e, gauss_window, mode='same')

        # **Step 3.4: 计算 FFT**
        N = len(x_e)
        fft_y = fft(z_e - np.mean(z_e))
        freqs = fftfreq(N, d=(x_e[1] - x_e[0]))
        fft_amplitude = 2 / N * np.abs(fft_y)

        # **Step 3.5: 找到所有峰值**
        peak_indices, _ = find_peaks(fft_amplitude[:N // 2], height=0.3 * np.max(fft_amplitude[:N // 2]))  # 仅保留较大峰值
        peak_freqs = freqs[peak_indices]  # 这些峰值对应的频率
        peak_amplitudes = fft_amplitude[peak_indices]  # 这些峰值的 FFT 振幅

        # **Step 3.6: 查找 0.28 和 0.56 附近的 peak**
        target_freqs = [0.28, 0.56]  # 目标频率
        freq_tolerance = 0.05  # 允许的误差范围 ±0.05/μm

        selected_freq = None
        selected_amplitude = None

        # **查找 0.28 和 0.56 附近的 peak**
        for target_freq in target_freqs:
            valid_indices = \
            np.where((peak_freqs >= target_freq - freq_tolerance) & (peak_freqs <= target_freq + freq_tolerance))[0]

            if len(valid_indices) > 0:
                max_idx = valid_indices[np.argmax(peak_amplitudes[valid_indices])]
                peak_freq = peak_freqs[max_idx]
                peak_amplitude = peak_amplitudes[max_idx]

                # 选择较大的 peak
                if selected_amplitude is None or peak_amplitude > selected_amplitude:
                    selected_freq = peak_freq
                    selected_amplitude = peak_amplitude

        # **如果 0.28 和 0.56 附近都没有 peak，则选择全局最大峰值**
        if selected_freq is None:
            dominant_freq_idx = np.argmax(fft_amplitude[:N // 2])  # 最高值索引
            selected_freq = freqs[dominant_freq_idx]
            selected_amplitude = fft_amplitude[dominant_freq_idx]
            log_print(f"⚠️ No peak near 0.28 or 0.56/μm, using global max frequency: {selected_freq:.4f} 1/μm")
        else:
            log_print(f"✅ Selected dominant frequency near target: {selected_freq:.4f} 1/μm")
        dominant_freq = selected_freq
        # **计算 Poling 周期**
        T = 1 / abs(selected_freq)
        if T > 30:
            log_print(f"⚠️ {data_file.name}: Computed Poling Period too large (2T={2 * T:.2f} μm), skipping...")
            continue

        # **Step 3.7: 画图保存 FFT 频谱，方便 debug**
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(freqs[:N // 2], fft_amplitude[:N // 2], color='g', label="FFT Spectrum")

        # **标记 0.28 和 0.56 处的候选峰值**
        for target_freq in target_freqs:
            ax.axvline(x=target_freq, color='b', linestyle="--", alpha=0.6, label=f"Target {target_freq:.2f} 1/μm")

        # **标记最终选择的频率**
        ax.axvline(x=selected_freq, color='r', linestyle="--", label=f"Selected Peak: {selected_freq:.2f} 1/μm")

        # **添加图例、标签**
        ax.set_xlabel("Spatial Frequency (1/μm)", size=14)
        ax.set_ylabel("FFT Amplitude", size=14)
        ax.set_title("FFT Spectrum - Peak Selection Debug")
        ax.legend()

        # **自动保存图片**
        fig.savefig(save_dir / f"{data_file.stem}_FFT_Debug.png",
                    dpi=600, transparent=True, bbox_inches='tight')

        # **关闭图像，释放内存**
        plt.close(fig)

        log_print(f"📊 FFT Debug Spectrum saved as {data_file.stem}_FFT_Debug.png")



        # **记录日志**
        log_print(f"Final Dominant Frequency: {dominant_freq:.4f} 1/μm")

        T = 1 / abs(dominant_freq)
        flip =1
        if T>2:
            T = T/2
            flip=0


        # **Step 3.4: 计算占空比**
        min_peak_distance = 0.5/ (x_e[1] - x_e[0])
        peaks, _ = find_peaks(1-z_e, distance=min_peak_distance,height=np.percentile(abs(1-z_e),50))
        # 检查是否找到峰值
        if len(peaks) < 2:
            print("Warning: No peaks found! Please check your input parameters or data.")
            continue  # 继续执行后续代码（适用于循环）
        duty_cycle = np.median(np.diff(peaks))*(x_e[1] - x_e[0]) / (2 * T) if len(peaks) > 1 else None


        if duty_cycle is not None:
            log_print(f"✅ {data_file.name} 处理完成: Poling Period 2T={2*T:.2f} μm, Duty Cycle={duty_cycle:.2f}")
        else:
            log_print(f"⚠️ {data_file.name} 处理完成: Poling Period 2T={2*T:.2f} μm, 但无法计算 Duty Cycle")

        # **Step 2: 让反转起点与 dark_peaks[0] 对齐**
        idx = np.argmin(z_e)
        x_shifted = x_e - x_e[idx]  # **平移 x 轴，使第一个 peak 对齐 0**

        # **Step 3: 计算 d_eff 反转**
        if flip:
            deff_signs = np.sign(np.sin(2 * np.pi / (2 * T) * x_shifted))  # 生成正弦反转信号
            z_e_reversed = z_e * deff_signs  # 反转 d_eff
            flip_regions = deff_signs < 0  # 负值区域
            # 绘制图像
            fig = plt.figure(figsize=(10, 5))
            plt.plot(x_e, z_e, label="Original $z_e$", linestyle='-', color='b')
            plt.plot(x_e, z_e_reversed, label="Reversed $z_e$", linestyle='--', color='r')

            # 颜色填充不同区域
            plt.fill_between(x_e, z_e.min(), z_e.max(), where=flip_regions, color='red', alpha=0.2,
                             label="Flipped Region")
            plt.fill_between(x_e, z_e.min(), z_e.max(), where=~flip_regions, color='blue', alpha=0.2,
                             label="Unflipped Region")

            # 图例和标签
            plt.xlabel("x")
            plt.ylabel("z_e values")
            plt.title("Comparison of $z_e$ and Reversed $z_e$")
            plt.legend()
            plt.grid(True)
            # **自动保存图片**
            fig.savefig(save_dir / f"{data_file.stem}_inverse_region.png",
                        dpi=600, transparent=True, bbox_inches='tight')
            plt.close(fig)

        else:
            z_e_reversed = z_e


        # **Step 3.5: 计算 QPM 结构**
        fft_y_qpm = fft(z_e_reversed - np.mean(z_e_reversed))  # 计算 FFT
        fft_amplitude_qpm = 2 / N * np.abs(fft_y_qpm)  # 归一化 FFT 幅度
        freqs_qpm = freqs[:N // 2]  # 取正频率部分

        # **检查最高频率是否在 0.298/μm 附近**
        qpm_peak_idx = np.argmax(fft_amplitude_qpm[1:N // 2])  # 最高值索引
        qpm_peak_freq = freqs_qpm[qpm_peak_idx+1]  # 最高频率
        qpm_peak_amplitude = fft_amplitude_qpm[qpm_peak_idx+1]  # 最高频率对应的 FFT 幅度

        # **如果最高值不在 0.298/μm 附近，则取 0.298/μm 处的值**
        target_freq = 0.298  # 目标频率
        target_idx = np.argmin(np.abs(freqs_qpm - target_freq))  # 找到最接近 0.28/μm 的索引
        target_amplitude = fft_amplitude_qpm[target_idx]  # 目标频率的 FFT 幅度

        if np.abs(qpm_peak_freq - target_freq) > 0.05:  # 允许 ±0.1/μm 误差
            log_print(f"⚠️ QPM Peak ({qpm_peak_freq:.4f} 1/μm) not near 0.298/μm, using 0.298/μm value.")
            qpm_peak_freq = target_freq
            qpm_peak_amplitude = target_amplitude

        # **日志记录**
        log_print(f"QPM Main Frequency: {qpm_peak_freq:.4f} 1/μm")
        log_print(f"QPM FFT Amplitude: {qpm_peak_amplitude:.4f}")

        # **Step 6: 绘制并保存 QPM 结构的 FFT 频谱**
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(freqs_qpm, fft_amplitude_qpm[:N // 2], color='g', label="QPM FFT Spectrum")

        # **标记最高频率**
        ax.axvline(x=qpm_peak_freq, color='r', linestyle="--",
                   label=f"Selected QPM Peak: {qpm_peak_freq:.2f} 1/μm")

        # **标记 0.28/μm 处的值**
        ax.axvline(x=target_freq, color='b', linestyle="--",
                   label="Target 0.28 1/μm", alpha=0.6)

        # **添加图例、标签**
        ax.set_xlabel("Spatial Frequency (1/μm)", size=14)
        ax.set_ylabel("FFT Amplitude", size=14)
        ax.set_title("Fourier Transform of Quasi-Phase-Matching (QPM) Structure")
        ax.legend()

        # **自动保存图片**
        fig.savefig(save_dir / f"{data_file.stem}_QPM_FFT.png",
                    dpi=600, transparent=True, bbox_inches='tight')

        # **关闭图像，释放内存**
        plt.close(fig)

        log_print(f"📊 QPM FFT Spectrum saved as {data_file.stem}_QPM_FFT.png")

    log_print(f"📁 所有文件处理完成，日志已保存至: {log_file_path.resolve()}")
