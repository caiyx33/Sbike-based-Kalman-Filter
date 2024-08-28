# Sbike-based-Kalman-Filter
# Neural Signal Processing and State Estimation

This repository contains MATLAB scripts for neural signal processing and state estimation using Kalman filters and other computational methods.

## Contents
- `PPAF.m`: Simulates a one-dimensional kinematic scenario with a triangular waveform modulated by a frequency change and computes the neural firing rate based on this modulation.
- `PPAF2.m`: Simulates neural signals with random walk kinematics and dynamic tuning of parameters, using a state estimation method.

## Prerequisites
- MATLAB (tested with MATLAB R2021a)
- Statistics and Machine Learning Toolbox

One-dimensional Kinematics Simulation: Navigate to the directory containing the script neural_signal_simulation_1.m and run it in MATLAB. This script will generate the neural spikes based on a modulated triangular waveform and plot the results.
Random Walk Kinematics Simulation: Open the script neural_signal_simulation_2.m in MATLAB and execute it. This will simulate a neural encoding model with random walk kinematics and apply a filtering algorithm to estimate the state.

Contributing
Feel free to fork this repository and submit pull requests to enhance the simulations or add new features.

# SBKF (State-Based Kalman Filter)

This MATLAB function implements a State-Based Kalman Filter (SBKF) for neural signal processing, designed to estimate the state of a system based on noisy observations and a model of the system's dynamics.

## Function Signature
```matlab
function X_estimated = SBKF(a, spikes, F, Q, dt, rho, alpha, beta)
Parameters
a : Initial state estimate.
spikes : Matrix of observed neural spikes, where rows correspond to different neurons and columns correspond to time steps.
F : State transition matrix.
Q : Process noise covariance matrix.
dt : Time step duration.
rho : Scaling factor for the firing rate.
alpha : Base firing rate.
beta : Tuning parameter matrix, affecting the modulation of the firing rate by the state.

To use this function, you need to define the initial conditions and parameters, and then call the function with these parameters. Here's an example of how to call the SBKF function:
% Example parameters
a = 0; % Initial state
N_t = 100; % Number of time steps
num_neurons = 5; % Number of neurons
spikes = rand(num_neurons, N_t) < 0.5; % Randomly generated spike data
F = 1; % Example state transition matrix
Q = 0.01; % Process noise covariance
dt = 0.01; % Time step
rho = 20; % Scaling factor for firing rate
alpha = -1; % Base firing rate
beta = repmat(0.5, num_neurons, N_t); % Example tuning parameter matrix

% Call the function
X_estimated = SBKF(a, spikes, F, Q, dt, rho, alpha, beta);
