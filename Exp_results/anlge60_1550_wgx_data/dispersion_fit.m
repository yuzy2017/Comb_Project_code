clear all;
load HongTang_1550_neff(1).mat

% 假设 mode1
beta_0 = real(beta(:,1)); % /m
c_const = 299792458;  %m/s
ng = c_const./vg(:,1);
wl = real(c_const./f);
omega = 2*pi.*f;
[~,idx_center]=min(abs(wl-1.55e-6));
% 对 beta_0 和 omega 进行多项式拟合
n =4;

x_centered = (wl - wl(idx_center)) / (max(wl) - min(wl));
p = polyfit(x_centered, ng, n);  % 这样先fit出来仿真得到的ng


% 求多项式的一阶、二阶、三阶导数
% p_prime_1 = polyder(p);            % 一阶导数
% p_prime_2 = polyder(p_prime_1);    % 二阶导数
% p_prime_3 = polyder(p_prime_2);    % 三阶导数
% we could use these derivative to calculate FSR but maybe not the most
% efficient way





L = 2*pi*90e-6;

% 打开文件
fid = fopen('w2_0.9_mode2_15dB_ attenuation.dat', 'r');

% 读取文件内容，跳过以 # 开头的注释行
data = textscan(fid, '%f %f', 'CommentStyle', '#');

% 关闭文件
fclose(fid);

% 将读取的数据转换为单独的变量
wavelength = data{1};  % 第一列: Wavelength
voltage = data{2};     % 第二列: Voltage

% 1. 归一化电压值
normalized_voltage = voltage / max(voltage);

% 2. 使用 islocalmin 找到局部最小值（下陷点）
min_idx = islocalmin(normalized_voltage,'MinSeparation',1.3/(wavelength(2)-wavelength(1)));

% 3. 获取下陷点对应的波长和电压值
dip_wavelengths = wavelength(min_idx);
dip_voltages = normalized_voltage(min_idx);

% 4. 绘制 voltage/max(voltage) 对 wavelength 的图
figure;
plot(wavelength, normalized_voltage, '-b', 'LineWidth', 1.5);
hold on;

% 在共振 dip 点标记位置
plot(dip_wavelengths, dip_voltages, 'ro', 'MarkerFaceColor', 'r');
xlabel('Wavelength (nm)');
ylabel('Normalized Voltage');
title('Voltage/Max(Voltage) vs Wavelength');
grid on;
legend('Normalized Voltage', 'Resonance Dips');


FSR = dip_wavelengths(2:end)-dip_wavelengths(1:end-1);
wavelength_center = (dip_wavelengths(1:end-1)+dip_wavelengths(2:end))/2;
wl_c = mean([max(wavelength_center),min(wavelength_center)]);

wl_rescale = (wavelength_center-wl_c)/(max(wavelength_center)-min(wavelength_center));
% 1. 计算FSR的均值和标准差
mean_FSR = mean(FSR);               % FSR的均值
std_FSR = std(FSR);                 % FSR的标准差

% 2. 设置一个过滤条件，移除超过3倍标准差的数据
filter = abs(FSR - mean_FSR) <= 3 * std_FSR;   % 只保留在3倍标准差内的数据

% 3. 对过滤后的数据进行polyfit拟合
wl_rescale_filtered = wl_rescale(filter);      % 过滤后的wl_rescale
FSR_filtered = FSR(filter);                   % 过滤后的FSR

% 4. 进行多项式拟合
p_FSR = polyfit(wl_rescale_filtered, FSR_filtered, 5);  
fitted_FSR = polyval(p_FSR, wl_rescale);
ng_sim = polyval(p,wl_rescale);

FSR_Sim = (wavelength_center*1e-9).^2./(ng_sim*L)*1e9;


figure;

% 绘制 FSR Raw
plot((dip_wavelengths(1:end-1)+dip_wavelengths(2:end))/2, FSR, 'DisplayName', 'FSR Raw', 'LineWidth', 1.5);
hold on;

% 绘制 FSR Fitted
plot(wavelength_center, fitted_FSR, 'DisplayName', 'FSR Fitted', 'LineWidth', 1.5);

% 绘制 FSR Sim
plot(wavelength_center, FSR_Sim, 'DisplayName', 'FSR Sim', 'LineWidth', 1.5);

% 添加图例
legend('Location', 'best', 'FontSize', 12);

% 添加标题和轴标签
title('FSR Comparison', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Wavelength (nm)', 'FontSize', 12);
ylabel('FSR (GHz)', 'FontSize', 12);

% 设置坐标轴字体大小
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

% 添加网格线
grid on;
hold off;


