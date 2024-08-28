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
