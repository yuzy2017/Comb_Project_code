clear all;

load xcut_comb1_4modes_ng.mat;
c=299792458;
% Create figure and set up axes
figure;
ax1 = axes('NextPlot', 'add', 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, ...
           'FontSize', 12, 'FontName', 'Arial');

% Plot data with enhanced line styles and colors
lineColors = lines(3);  % Use distinct colors for each line
plot(lum.x0, lum.y0, 'Color', lineColors(1, :), 'LineWidth', 1.5, 'DisplayName', 'TE0');
hold on;
plot(lum.x1, lum.y1, 'Color', lineColors(2, :), 'LineWidth', 1.5, 'DisplayName', 'TE1');
plot(lum.x2, lum.y2, 'Color', lineColors(3, :), 'LineWidth', 1.5, 'DisplayName', 'TM0');
plot(lum.x3, lum.y3, 'Color', lineColors(3, :), 'LineWidth', 1.5, 'DisplayName', 'TE2');
% Set axis limits
set(ax1, 'XLim', [1.39 1.65], 'YLim', [2.35 2.43]);

% Label axes
xlabel('Wavelength (\mum)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Group Index (n_g)', 'FontSize', 14, 'FontWeight', 'bold');

% Customize legend
legend('show', 'FontSize', 12, 'Location', 'best', 'Box', 'off');

% Set figure background color
set(gcf, 'Color', 'w');

% Optional: add title
title('Group Index vs. Wavelength', 'FontSize', 14, 'FontWeight', 'bold');
freq_more = linspace(c./max(lum.x0),c./min(lum.x0),651); %MHz lum.x in um
% interpolerate to get more points
wl_more =c./freq_more;

%ng_more =interp1(lum.x0,lum.y0,wl_more,'spline');
ng_more =interp1(lum.x3,lum.y3,wl_more,'spline');
L = 2.4226e+03;




FSR = c./(ng_more*L);  %  unit GHz

%[~,wl_cent] = max(ng_more);
wl_pump = 1.5519715;
[~,wl_cent] = min(abs(wl_more-wl_pump));

FSR0 = FSR(wl_cent);
FSR_diff = FSR-FSR0;

Dint = cumsum(FSR_diff);

Dint = Dint-Dint(wl_cent);
%plotting
figure;

ax1 = gca;
plot(ax1,freq_more,Dint,'LineWidth',1.5)
xlabel(ax1, 'Frequency Offset from Pump (THz)');
ylabel(ax1, 'Dint');
%title('Integrated Dispersion Dint vs Frequency Offset');
grid on;

%Adding a second x-axis for wavelength
ax2 = axes('Position', ax1.Position, 'XAxisLocation', 'top', 'YAxisLocation', 'right', ...
           'Color', 'none', 'XColor', 'k', 'YColor', 'none');
set(ax2, 'XLim', ax1.XLim);  % Match x-limits to primary axis

%set the ticks
wavelength_ticks= 1.35:0.05:1.70;
freq_ticks = c./wavelength_ticks;


set(ax2, 'XTick', flip(freq_ticks), 'XTickLabel', flip(round(wavelength_ticks, 2))); % Flip to show large to small
xlabel(ax2, 'Wavelength (nm)');

% Adjust the title position upwards to avoid overlap with the top axis
title('Integrated Dispersion Dint vs Frequency Offset', 'Units', 'normalized', 'Position', [0.5, 1.05, 0]); 
brush on;


figure;

ax1 = gca;
plot(ax1,wl_more,Dint,'LineWidth',1.5)
xlabel(ax1, 'Wavelength (nm)');
ylabel(ax1, 'Dint');
title('Integrated Dispersion Dint vs Frequency Offset');
grid on;

%title('Integrated Dispersion Dint vs Frequency Offset');