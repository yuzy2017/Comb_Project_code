% Specify the filename
filename = 'W0000.csv';

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
% for 13 could be a soliton
% Convert power from dB to amplitude
amplitude = 10.^(power_db / 20);

% Define the sec^2 function for fitting
sec2_func = @(params, x) params(1) * sech(params(2) * (x - params(3))).^2;

% Initial guesses for parameters [A, B, C]
wl_teeth = [1549.14,1549.13,1550.35,1551.56,1554,1555.23,1556.45,1557.68];
ampl = [-49.834,-49.898,-44.759,-30.15,-22.721,-31.467,-40.8,-46.148];
ampl = 10.^(ampl/10);
scale_factor = max(ampl);
ampl = ampl/scale_factor;
initial_params = [max(ampl), 0.01, 1552.77];
figure 
plot(wl_teeth,ampl,'linewidth',2)


% Perform the fitting using non-linear least squares
params_opt = lsqcurvefit(sec2_func, initial_params, wl_teeth, ampl);


%for example 
%params_opt(3) = 1552.77;
% Generate fitted curve
fitted_amplitude = sec2_func(params_opt, wavelength)*scale_factor;

% Convert fitted amplitude back to dB
fitted_power_db = 10 * log10(fitted_amplitude);

% Plot the original data and the fit
figure;
plot(wavelength, power_db, '-b', 'DisplayName', 'Original Data');
hold on;
plot(wavelength, fitted_power_db, '-r', 'LineWidth', 1.5, 'DisplayName', 'Sech^2 Fit');
xlabel('Wavelength (nm)');
ylabel('Power (dB)');
title('Optical Spectrum Analyzer Trace with Sec^2 Fit');
legend;
grid on;
hold off;
ylim([-70,20])


% Define constants and parameters
lambda_0 = 1550e-9;  % Pump wavelength in meters (e.g., 1550 nm)
P0 = 1;              % Pump power in Watts
gamma = 10;           % Nonlinearity coefficient (1/W/m)
beta2 = -10e-27;     % GVD parameter in s^2/m (example value)

% Define wavelength range around the pump wavelength
lambda = linspace(1450e-9, 1650e-9, 1000);  % Wavelength range in meters

% Calculate frequency detuning omega for each wavelength
omega = 2 * pi * 3e8 * (1 ./ lambda - 1 / lambda_0);  % in rad/s

% Calculate MI gain using the formula
G = 2 * gamma * P0 * (sqrt(1 - (beta2 * omega.^2) / (2 * gamma * P0)) - 1);

% Plot MI gain vs wavelength
figure;
plot(lambda * 1e9, G, '-b');  % Convert lambda to nm for x-axis
xlabel('Wavelength (nm)');
ylabel('MI Gain');
title('Modulation Instability Gain vs. Wavelength');
grid on;
