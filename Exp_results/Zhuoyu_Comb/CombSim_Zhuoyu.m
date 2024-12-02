clear all
close all

hbar = 1.054e-34;
c = 299792458;
Nspan = 128;
Ntot = 2*Nspan; %here maybe try even number of modes later
mode_idx = linspace(-Nspan,Nspan-1,Ntot)';
mode_idx = fftshift(mode_idx);

% define dispersion
d1a = 4.5665e12;
d1b = 4.4189e12;
d1c = 4.25e12;

d2a = 2*8.84e8;
d2b = 2*4.31e8;
d2c = 2*2.31e8;

d3a = 2*pi*153e3;
d3b = -2*pi*103e3;

d4a = -2*pi*3.3e3;


% here is very tricky; maybe assume resonance, phase matching very strong
% restriction‘

% here just introduce the phase matching wavelength at the center of the
% corresponding 
% here is the phase mismatch parameter and I did not know how to compensate
% it yet 
% here we just do not introduce poling first
nboff = 40;
ncoff = 25;

omega_0a = 2*pi*c/(1550e-9);
%omega_0b = 2*omega_0a+(2*nboff*d1b+4*d2b/2*nboff^2)-2*(nboff*d1a+d2a/2*nboff);
omega_0b = 2*omega_0a+(nboff*d1b+d2b/2*nboff^2)-(nboff*d1a+d2a/2*nboff);
%omega_0c = 3*omega_0a+(3*nboff*d1b+9*d2b/2*nboff^2)-3*(nboff*d1a+d2a/2*nboff);
omega_0c = 3*omega_0a+(nboff*d1b+d2b/2*nboff^2)-(nboff*d1a+d2a/2*nboff);

%do I need to write them into a function???
omega_a = omega_0a + d1a*mode_idx + d2a/2.*mode_idx.^2;
omega_b = omega_0b + d1b*mode_idx + d2b/2.*mode_idx.^2;
omega_c = omega_0c + d1c*mode_idx + d2c/2.*mode_idx.^2;

% from Q factor calculate the line width kappa
% or you can directly read it from your spectrum 

kappa_a = omega_0a/(2*500*1e3);
kappa_a1 = kappa_a/2; kappa_a0 = kappa_a/2; % this for power input

kappa_b = omega_0b/(2*70*1e3);
kappa_b1 = kappa_b/3; kappa_b0 = kappa_b*2/3;

kappa_c = omega_0c/(2*70*1e3);
kappa_c1 = kappa_c/2; kappa_c0 = kappa_c/2;

omega_r = 2*pi*18.08e12;
kappa_r = 2*pi*60*1e9;

% or here I can directly draw the frequency mismatch -N/2
% to N/2
z = linspace (-Nspan/2+1,Nspan/2-1,2*(Nspan/2-1)+1);
figure;
plot(2*pi*c./omega_a(Nspan+z)*1e9,(omega_b(Nspan+2*z)-2*omega_a(Nspan+z))/kappa_a)
grid on 


%introducing nonlinear coupling terms
g2aab = 2*pi*0.08*1.0e6*1.00;
g2abc = 2*pi*0.08*1.0*10^6*0.10;
g3aa = -2*pi*2.8*1.00;
g3bb = -2*pi*14.9*1.00;
g3ab = -2*pi*17.1*1.00;
g3ac = -2*pi*0.1*1.00;
gR = 2 *pi*0.0003*10^6;

% pump parameters
p = 0; % pump mode number
Pin = 2.9*1; % W pump power
Ein = sqrt(2*kappa_a1*Pin/(hbar*omega_0a));
delta0 = -15;
delta = delta0*kappa_a;
% initial cavity field
EA = 10.^(-10-10*rand(Ntot,1));
EA0 = EA;
EB = EA;
EC = EA;
ER = EA;

omega_p = omega_0a + p*d1a+d2a/2*p^2+delta;
delta_a = d2a/2*(mode_idx.^2-p^2)-delta;
delta_b = omega_0b+d1b*mode_idx+d2b/2*mode_idx.^2-2*omega_p-(mode_idx-2*p)*d1a;
delta_c = omega_0c+d1c*mode_idx+d2c/2*mode_idx.^2-3*omega_p-(mode_idx-3*p)*d1a;
delta_r = sign(omega_r - mode_idx*d1a).*min(abs(omega_r-mode_idx*d1a),30*kappa_r);




%create pump vector
kappa_norm =kappa_a;
Anorm = 1e3;

Pump = zeros(Ntot,1);
%Pump(Nspan+p) = Ein/(kappa_norm*Anorm);
Pump(p+1) = Ein/(kappa_norm*Anorm);
% build the linear evolution matrix
MCa = (-1i*delta_a-kappa_a)/kappa_norm;
MCb = (-1i*delta_b-kappa_b)/kappa_norm;
MCc = (-1i*delta_c-kappa_c)/kappa_norm;
MCr = (-1i*delta_r-kappa_r)/kappa_norm;

IDD = eye(Ntot);
MCI = ones(Ntot,1);

%normalize the interaction parameter
G2aab = -2i*g2aab*sqrt(Ntot)*Anorm/kappa_norm*MCI+EA0;
G2abc = -1i*g2abc*sqrt(Ntot)*Anorm/kappa_norm*MCI+EA0;
G3aa = -2i*g3aa*Ntot *Anorm^2/kappa_norm*MCI+EA0;
G3ab = -1i*g3ab*Ntot*Anorm^2/kappa_norm*MCI+EA0;
G3bb = -2i*g3bb*Ntot*Anorm^2/kappa_norm*MCI+EA0;
G3ac = -1i*g3ac*Ntot*Anorm^2/kappa_norm*MCI+EA0;
GR = -1i*gR*sqrt(Ntot)*Anorm^2/kappa_norm*MCI+EA0;

%time scale 
nanosec = 1e-9;
picosec = 1e-12;
T = 4.5*nanosec*kappa_norm;
%time step
dt = 0.008*picosec*kappa_norm;Ndt=round(0.01*T/dt);


% here start simulation????
% ODE function with decay term
% just model other parameters as constant and put it into the function.
function dydt = nls_decay_ode(t, y, Ntot,Pump, G2aab,G2abc,G3aa,G3ab,G3bb,G3ac,GR, MCa,MCb,MCc,MCr)
    EA = y(1:Ntot);
    EB = y(Ntot+1:2*Ntot);
    EC = y(2*Ntot+1:3*Ntot);
    ER = y(3*Ntot+1:end);

    Fa = fft(EA)/sqrt(Ntot);
    Fb = fft(EB)/sqrt(Ntot);
    Fc = fft(EC)/sqrt(Ntot);
    Fr = fft(ER)/sqrt(Ntot);

    TWMa = ifft(conj(Fa).*Fb)*sqrt(Ntot);
    TWMb = ifft(Fa.^2)*sqrt(Ntot);

    FWMaa = ifft(Fa.^2.*conj(Fa))*sqrt(Ntot);
    FWMab = ifft(Fa.*conj(Fb).*Fb)*sqrt(Ntot);
    FWMbb = ifft(Fb.^2.*conj(Fb))*sqrt(Ntot);
    FWMba = ifft(Fa.*conj(Fa).*Fb)*sqrt(Ntot);
    % for interaction with higher frequency
    TWMac = ifft(conj(Fb).*Fc)*sqrt(Ntot);
    TWMabc = ifft(Fa.*Fb)*sqrt(Ntot);
    FWMac = ifft(conj(Fa).^2.*Fc)*sqrt(Ntot);
    FWMaaa = ifft(Fa.^3)*sqrt(Ntot);
    TWMbc = ifft(conj(Fa).*Fc)*sqrt(Ntot);

    % I am not sure about the raman process but we can check
    MRa = ifft(Fa.*Fr)*sqrt(Ntot);
    MRaconj = ifft(Fa.*conj(Fr)); % I am not sure if writing this is right
    MRaa = ifft(conj(Fa).*Fa)*sqrt(Ntot);
    


    % Here we only check ab first I did not care about c at this time
    da_dt = MCa.*EA + G2aab.*TWMa + G3aa.*FWMaa + G3ab.*FWMab+GR.*MRa+GR.*MRaconj+ G2abc.*TWMac+G3ac.*FWMac +  Pump;
    db_dt = MCb.*EB + G2aab.*TWMb + G3bb.*FWMbb + G3ab.*FWMba +G2abc.*TWMbc;
    dc_dt = MCc.*EC +G2abc.*TWMabc +G3ac.*FWMaaa;
    dR_dt = MCr.*ER + GR.*MRaa;

    % Return the time derivatives
    dydt = [da_dt; db_dt; dc_dt; dR_dt];
end
% maybe I will need parameters in this problem
% how pycharm code write to get the matrix?
% Initial condition vector
y0 = [EA; EB; EC; ER];

% Time span for the solution
tspan = 0:dt:T;

% Solve the system of ODEs using ode45
[t, y] = ode45(@(t, y) nls_decay_ode(t, y, Ntot,Pump, G2aab,G2abc,G3aa,G3ab,G3bb,G3ac,GR, MCa,MCb,MCc,MCr), tspan, y0);

% then we draw the evolution we could get the comb structure and try to see
% how we can play around this code.

% Extract solution (real and imaginary parts)
EA_result = y(:, 1:Ntot);  % fundamental field
EB_result = y(:, Ntot+1:2*Ntot);  % double frequency
EC_result = y(:,2*Ntot+1:3*Ntot); % triple frequency

% I am not sure why they rearrange the frequency here
% Define the shift amount
%shift_amount = -(Nspan + 1);
%I think we do not need shifts
shift_amount = Nspan+1;

% Rearrange the frequencies using circshift
li_Omega_a = fftshift(omega_a);
li_Omega_b = fftshift(omega_b);
li_Omega_c = fftshift(omega_c);
% here get the wavelength and the power of each mode but I do not know why
% they need this??? still confusing
% Vectorized wavelength calculation for each mode (in nm)
CombA(:, 1) = (2 * pi * 3e8) ./ li_Omega_a * 1e9;
CombB(:, 1) = (2 * pi * 3e8) ./ li_Omega_b * 1e9;
CombC(:, 1) = (2 * pi * 3e8) ./ li_Omega_c * 1e9;

% Vectorized power calculation in dB for each mode
CombA(:, 2) = 10 * log10(hbar * omega_0a * 2 * kappa_a1 * abs(EA).^2 * abs(Anorm)^2 + 1e-199);
CombB(:, 2) = 10 * log10(hbar * omega_0b * 2 * kappa_b1 * abs(EB).^2 * abs(Anorm)^2 + 1e-199);
CombC(:, 2) = 10 * log10(hbar * omega_0c * 2 * kappa_b1 * abs(EC).^2 * abs(Anorm)^2 + 1e-199);

% output power
CombAav = abs(EA_result).^2;
CombBav = abs(EB_result).^2;
CombCav = abs(EC_result).^2;

PulseLength = 40;

PAtot=sum(hbar*omega_0a*2*kappa_a1*abs(Anorm)^2*CombAav,2)/PulseLength;
PBtot=sum(hbar*omega_0b*2*kappa_b1*abs(Anorm)^2*CombBav,2)/PulseLength;
PCtot=sum(hbar*omega_0c*2*kappa_c1*abs(Anorm)^2*CombCav,2)/PulseLength;

% Define time steps and initialize GIF parameters
wait_t = 0.1; % Time step for updating the plot
gif_filename = 'comb_simulation.gif'; % Filename for GIF
is_save_gif = true; % Set to true if you want to save GIF

% Predefined axis limits and ticks for consistency
freq_ticksA = linspace(1450, 1650, 5);  % X-axis ticks for CombA
freq_ticksB = linspace(700, 850, 5);    % X-axis ticks for CombB
power_ticks = -100:20:0;                % Y-axis ticks for power plots
time_ticks = linspace(0, T, 5);         % X-axis ticks for time plots
output_power_ticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0];  % Y-axis ticks for log scale

% Time loop for simulation
figure;

for k = 1:Ndt/5:length(t)
    % Extract frequency and power data at time t(k)
    CombA_current_freq = CombA(:, 1);
    CombB_current_freq = CombB(:, 1);
    CombA_current_power = fftshift(CombAav(k,:));
    CombB_current_power = fftshift(CombBav(k, :));
    
    % Calculate total output power at time t(k)
    PAtot_current = PAtot(k);
    PBtot_current = PBtot(k);

    % Clear the figure for new plots
    clf;

    % Plot CombA frequency spectrum
    subplot(2, 2, 1);
    plot(CombA_current_freq, 10 * log10(hbar * omega_0a * 2 * kappa_a1 * CombA_current_power * abs(Anorm)^2 + 1e-199), 'b');
    xlabel('Frequency (GHz)');
    ylabel('Power (dB)');
    title(['CombA Spectrum at t = ' num2str(t(k), '%.2f') ' ns']);
    grid on;
    ylim([-100,0]);
    xlim([1450,1650]);
    set(gca, 'XTick', freq_ticksA, 'YTick', power_ticks); % Fix ticks

    % Plot CombB frequency spectrum
    subplot(2, 2, 2);
    plot(CombB_current_freq, 10 * log10(hbar * omega_0b * 2 * kappa_b1 * CombB_current_power * abs(Anorm)^2 + 1e-199), 'r');
    xlabel('Frequency (GHz)');
    ylabel('Power (dB)');
    title(['CombB Spectrum at t = ' num2str(t(k), '%.2f') ' ns']);
    grid on;
    ylim([-100,0]);
    xlim([700,850]);
    set(gca, 'XTick', freq_ticksB, 'YTick', power_ticks); % Fix ticks

    % Plot PAtot over time
    subplot(2, 2, 3);
    plot(t(1:k), PAtot(1:k), 'b');
    xlabel('Time (s)');
    ylabel('Output Power PAtot (W)');
    title('PAtot over Time');
    set(gca, 'YScale', 'log', 'YTick', output_power_ticks, 'XTick', time_ticks); % Fix ticks
    grid on;
    ylim([1e-10,10]);

    % Plot PBtot over time
    subplot(2, 2, 4);
    plot(t(1:k), PBtot(1:k), 'r');
    xlabel('Time (s)');
    ylabel('Output Power PBtot (W)');
    title('PBtot over Time');
    set(gca, 'YScale', 'log', 'YTick', output_power_ticks, 'XTick', time_ticks); % Fix ticks
    grid on;
    ylim([1e-10,10]);

    % Update the figure
    drawnow;

    % Capture the frame and save to GIF
    if is_save_gif
        frame = getframe(gcf);
        img = frame2im(frame);
        [imind, cm] = rgb2ind(img, 256);
        if k == 1
            imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
    
    % Pause to control the update rate (optional)
    pause(wait_t); % Adjust pause time for desired animation speed
end
