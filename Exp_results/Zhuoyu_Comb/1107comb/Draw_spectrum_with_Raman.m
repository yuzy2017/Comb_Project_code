% Specify the filename
filename = 'W0003.csv';

% Read the data, skipping rows until the actual data starts
data = readtable(filename, 'FileType', 'text', 'HeaderLines', 26);  % Adjust 'HeaderLines' if needed

% Extract the wavelength and power columns
wavelength = data{:, 1};  % First column for wavelength
power_db = data{:, 2};    % Second column for power in dB
[~,peak_idx]=max(power_db);
% Plot the data
figure;
hold on;

% Plot the main optical spectrum (in blue with a thicker line)
plot(wavelength, power_db, '-b', 'LineWidth', 2);

% Parameters for the Raman scattering lines (center in nm, FWHM in nm)
raman_peaks = [
    1494.1, 12.03;  % First peak at 1494.1 nm with bandwidth 12.03 nm
    1479.29, 5.18; % Second peak at 1479.29 nm with bandwidth 5.18 nm
    1425.57, 7.78;
    1415, 7.75; 
    1367.3, 6.09     % Third peak at 1415 nm with bandwidth 7.75 nm
];

% Define the amplitude for the Raman lines (this can be adjusted)
A = [10,3,3,12,10];  % You can adjust this value to scale the Raman lines to the desired height

% Loop through the Raman peaks and add each Gaussian line
for i = 1:size(raman_peaks, 1)
        mu = raman_peaks(i, 1);  % Center of the peak
        FWHM = raman_peaks(i, 2); % Full width at half maximum
        sigma = FWHM / (2 * sqrt(2 * log(2))); % Convert FWHM to standard deviation
    
        % Gaussian function for the Raman peak
        raman_line = A(i) * exp(-((wavelength - mu).^2) / (2 * sigma^2))-65;
    
        % Plot the Raman line with a dashed red line and thicker line width
        plot(wavelength(1:peak_idx), raman_line(1:peak_idx), '--r', 'LineWidth', 1.5);

end
stokes_peaks =[1616.22,12.03;1633.9,5.18];
A_stokes = [11,3.3];
% Loop through the Raman peaks and add each Gaussian line
for i = 1:size(stokes_peaks, 1)
        mu = stokes_peaks(i, 1);  % Center of the peak
        FWHM = stokes_peaks(i, 2); % Full width at half maximum
        sigma = FWHM / (2 * sqrt(2 * log(2))); % Convert FWHM to standard deviation
    
        % Gaussian function for the Raman peak
        raman_line = A_stokes(i) * exp(-((wavelength - mu).^2) / (2 * sigma^2))-57;
    
        % Plot the Raman line with a dashed red line and thicker line width
        plot(wavelength(peak_idx:end), raman_line(peak_idx:end), '--r', 'LineWidth', 1.5);

end

% Add labels and title with improved formatting
xlabel('Wavelength (nm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Power (dBm)', 'FontSize', 14, 'FontWeight', 'bold');
title('Optical Spectrum with Raman Scattering Peaks', 'FontSize', 16, 'FontWeight', 'bold');

% Adjust plot limits and grid
xlim([1350, 1700]);
ylim([-70, 20]);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);  % Light grid lines for better visibility

% Add a legend
legend('Optical Spectrum', 'Raman E(TO3) 1494.1 nm', 'Raman A(TO3)  1479.29 nm', 'Raman E(TO6) 1425.51 nm', 'Raman A(TO4) 1415 nm','Raman A(LO4) 1367 nm',...
       'Location', 'northeast', 'FontSize', 12, 'Box', 'off');

% Make the figure background white and improve the axis appearance
set(gca, 'Color', 'w', 'FontSize', 12, 'LineWidth', 1.5);
set(gcf, 'Color', 'w');  % Set figure background to white

% Hold off to stop adding more elements to the plot
hold off;

