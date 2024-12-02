% Specify the filename
filename = 'W0011.csv';

% Read the data, skipping rows until the actual data starts
data = readtable(filename, 'FileType', 'text', 'HeaderLines', 26);  % Adjust 'HeaderLines' if needed

% Extract the wavelength and power columns
wavelength = data{:, 1};  % First column for wavelength
power_db = data{:, 2};    % Second column for power in dB

% Plot the data
figure;
plot(wavelength, power_db, '-b');
xlabel('Wavelength (nm)');
ylabel('Power (dBm)');
title('Optical Spectrum Analyzer Trace');
grid on;
ylim([-70,20]);

% here we try to calculate the FSR of the three modes

wl_idx = (wavelength>1555)&(wavelength<1557);
wl_comb = wavelength(wl_idx);
voltage_comb = power_db(wl_idx);

[peak_v, peak_wl] = findpeaks(voltage_comb+60,wl_comb,'MinPeakProminence', -51+60)
peak_v = peak_v - 60;

[dip_v,dip_wl] = findpeaks (-voltage_comb,wl_comb,'MinPeakHeight',55);

% 绘制光谱并标记峰值和谷值
figure;
plot(wl_comb, voltage_comb, '-', 'LineWidth', 1.5, 'Color', [0.1, 0.5, 0.8]);  % 原始光谱，蓝色线条
hold on;
plot(peak_wl, peak_v, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);  % 红色实心圆标记峰值
plot(dip_wl, -dip_v, 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 6);   % 绿色实心方块标记谷值
xlabel('Wavelength (nm)', 'FontSize', 12);
ylabel('Intensity (dB)', 'FontSize', 12);
title('Comb Spectrum with Peak and Dip Detection', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Spectrum', 'Peaks', 'Dips', 'Location', 'best');
set(gca, 'FontSize', 11);

% 计算并绘制不同模式的梳齿间距
Teeth_dist = peak_wl(2:end) - peak_wl(1:end-1);
mode_dist2 = dip_wl(3:2:end-1) - dip_wl(1:2:end-3);
mode_dist3 = dip_wl(4:2:end) - dip_wl(2:2:end-2);

% 绘制梳齿间距图
figure;
hold on;
plot(peak_wl(1:end-1), Teeth_dist, '-o', 'LineWidth', 1.5, 'Color', [0.8, 0.3, 0.3], 'MarkerFaceColor', [0.8, 0.3, 0.3], 'MarkerSize', 5);  % 红色曲线表示主要模式
plot(dip_wl(1:2:end-3), mode_dist2, '-s', 'LineWidth', 1.5, 'Color', [0.3, 0.7, 0.4], 'MarkerFaceColor', [0.3, 0.7, 0.4], 'MarkerSize', 5);  % 绿色方块表示模式2
plot(dip_wl(2:2:end-2), mode_dist3, '-^', 'LineWidth', 1.5, 'Color', [0.2, 0.4, 0.9], 'MarkerFaceColor', [0.2, 0.4, 0.9], 'MarkerSize', 5);  % 蓝色三角形表示模式3
xlim([1555,1557])
ylim([0.36,0.42])
% 设置图表美化细节
xlabel('Wavelength (nm)', 'FontSize', 12);
ylabel('Teeth Distance (nm)', 'FontSize', 12);
title('Teeth Distance for Different Modes', 'FontSize', 14, 'FontWeight', 'bold');
legend('Main Mode', 'Mode 2', 'Mode 3', 'Location', 'best');
grid on;
set(gca, 'FontSize', 11);

wl_idx = (wavelength>1548.67)&(wavelength<1558);
wl_comb = wavelength(wl_idx);
voltage_comb = power_db(wl_idx);

[peak_v, peak_wl] = findpeaks(voltage_comb+60,wl_comb,'MinPeakProminence', -51+60)
peak_v = peak_v - 60;
% 绘制梳状光谱，并标记峰值
figure;
plot(wl_comb, voltage_comb, '-', 'LineWidth', 1.5, 'Color', [0.1, 0.5, 0.8]);  % 原始光谱，线条加粗并设为蓝色
hold on;
plot(peak_wl, peak_v, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);  % 红色圆点标记峰值
xlabel('Wavelength (nm)', 'FontSize', 12);
ylabel('Intensity (dB)', 'FontSize', 12);
title('Comb Spectrum with Peak Detection', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Spectrum', 'Detected Peaks', 'Location', 'best');
set(gca, 'FontSize', 11);

% 计算和绘制梳齿间距
Teeth_dist = peak_wl(2:end) - peak_wl(1:end-1);

figure;
plot(peak_wl(1:end-1), Teeth_dist, '-o', 'LineWidth', 1.5, 'Color', [0.8, 0.3, 0.3], 'MarkerFaceColor', [0.8, 0.3, 0.3]);
xlabel('Wavelength (nm)', 'FontSize', 12);
ylabel('Teeth Distance (nm)', 'FontSize', 12);
title('Teeth Distance Between Peaks', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);
