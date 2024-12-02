% this one I think more or less the same as the kerr comb but the frequency
% spectrum is not as expected to be a comb.
% I think maybe the parameter need to be changed but I think the equations
% are good
clear all


% Parameters
L = 50;          % Length of the domain
Nx = 512;         % Number of spatial grid points
dx = L / Nx;      % Spatial step size
dt = 0.001;        % Time step size
Tmax = 1;        % Maximum simulation time
gamma = 0.2;        % Nonlinearity coefficient (change as needed)
decay = 13;      % Decay coefficient (adjust for damping)

% Spatial grid
x = linspace(-L/2, L/2, Nx);

% Initial condition (Gaussian pulse)
psi0 = exp(-x.^2);

% Rewrite the NLSE in terms of real and imaginary parts
u0 = real(psi0);  % Real part
v0 = imag(psi0);  % Imaginary part

% System of first-order ODEs: [u, v] = [Re(psi), Im(psi)]
% du/dt = -0.5 * d^2 u/dx^2 + gamma * u^3 - 3 * gamma * u * v^2 - decay * u
% dv/dt = -0.5 * d^2 v/dx^2 + gamma * v^3 - 3 * gamma * v * u^2 - decay * v

% ODE function with decay term
function dydt = nls_decay_ode(t, y, Nx, dx, gamma, decay)
    u = y(1:Nx);
    v = y(Nx+1:end);
    
    % Compute second derivatives (Laplace operator in 1D)
    d2u = (circshift(u, -1) - 2*u + circshift(u, 1)) / dx^2;
    d2v = (circshift(v, -1) - 2*v + circshift(v, 1)) / dx^2;
    
    % Compute the derivatives of u and v, including the decay term
    du_dt = -0.1 * d2u + gamma * (u.^3 - 3 * u .* v.^2) - decay * u;
    dv_dt = -0.1 * d2v + gamma * (v.^3 - 3 * v .* u.^2) - decay * v;
    
    % Return the time derivatives
    dydt = [du_dt; dv_dt];
end

% Initial condition vector
y0 = [u0; v0];

% Time span for the solution
tspan = [0 Tmax];

% Solve the system of ODEs using ode45
[t, y] = ode45(@(t, y) nls_decay_ode(t, y, Nx, dx, gamma, decay), tspan, y0);

% Extract solution (real and imaginary parts)
u = y(:, 1:Nx);  % Real part of psi
v = y(:, Nx+1:end);  % Imaginary part of psi

% Create a figure for plotting time and frequency dynamics
figure;

% Loop through time steps and plot both time and frequency domains
for i = 1:length(t)
    % Plot time domain: |ψ(x,t)|^2 (intensity)
    subplot(2, 1, 1);  % Upper half of the figure
    plot(x, fftshift(u(i, :).^2 + v(i, :).^2), 'LineWidth', 2);
    xlabel('x');
    ylabel('|\psi(x,t)|^2');
    title(['Time Domain: Time = ', num2str(t(i))]);
    axis([-L/2 L/2 0 1.5*max(u(i, :).^2 + v(i, :).^2)]);
    grid on;

    % Compute the Fourier transform of the wavefunction at time t(i)
    psi = u(i, :) + 1i * v(i, :);  % Combine real and imaginary parts into complex psi
    psi_freq = fftshift(fft(psi));  % Perform Fourier Transform and shift the zero frequency to center
    f = fftshift((0:Nx-1) - Nx/2) / L;

    % Plot frequency domain: |ψ(k,t)|^2 (intensity in frequency space)
    subplot(2, 1, 2);  % Lower half of the figure
    plot(f, abs(psi_freq).^2, 'LineWidth', 2); % Plot |ψ(k,t)|^2
    xlabel('Frequency (k)');
    ylabel('|\psi(k,t)|^2');
    title('Frequency Domain');
    axis([-max(f) max(f) 0 1.5*max(abs(psi_freq).^2)]);
    grid on;

    % Pause to update the figure in real-time
    drawnow;
end