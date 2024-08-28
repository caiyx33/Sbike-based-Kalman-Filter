%% 初始化
clc;
clear;
close all;

%% 仿真神经信号
%运动参数设置
dt = 0.01; % 时间步长，10ms
total_time = 10; % 总时间，10秒
N_t = total_time / dt; % 计算总的时间步数
T=1:dt:total_time-dt; %生成时间序列
n_r = 2.5e-3; % 随机游走幅度
n_v = 2.5e-5; % 噪声方差
% 神经元参数设置
num_neurons = 5;
rho = 20; % 神经元的ρ
alpha = -1; % 所有神经元的α
beta = [0.5,-0.5,0.5,-0.5,0.5]'*ones(1,N_t); % 调谐参数β
% 生成随机游走的运动学序列
x = zeros(1, N_t);% 初始化运动学序列
for t = 2:N_t
    x(t) = x(t-1) + sqrt(n_r)*randn;
end
x = x + sqrt(n_v)*randn(1,N_t);
% 对第一个神经元的β进行调节
beta_change_time1 = 3 / dt; % 30ms时调节β
beta_change_time2 = 7 / dt; % 70ms时调节β
beta(1, 1:beta_change_time1) = 0.5;
beta(1, beta_change_time1+1:beta_change_time2) = 0.5;
beta(1, beta_change_time2+1:end) = 0.5;
% 计算每个神经元在每个时间步的TFP
lambda = zeros(num_neurons, N_t);
for i = 1:num_neurons
    lambda(i, :) = rho * exp(alpha + beta(i, :) .* x);
end
% 生成脉冲列
spikes = rand(num_neurons, N_t) < lambda * dt;

%% 绘制仿真神经信号与偏好方向
encoding_test(dt,x,lambda,spikes,beta);

% 参数初始化
M=10;
x_est = zeros(M, N_t); % 初始化状态估计
W_post = 0.01; % 初始化方差估计，假设为1

% 状态转移和测量噪声参数
F = 1;  % 状态转移矩阵，假设系统稳定
Q = 0.005;  % 过程噪声方差

% 运行滤波算法
for m=1:M
    % 生成脉冲列
    for k = 2:N_t
        % 预测步
        x_pri = F * x_est(m,k-1);
        W_pri = F * W_post * F' + Q;
        % 计算lambda
        lambda_pri = rho * exp(alpha + beta(:,k)' * x_pri); % 计算尖峰率
        log_lambda = log(rho) + alpha + beta(:,k)' * x_pri;
        d_log_lambda = beta(:,k)';  % log(lambda) 对 theta 的一阶导数
        dd_log_lambda = 0;          % log(lambda) 对 theta 的二阶导数为 0
        % 更新方差和状态
        temp3=0;temp4=0;
        for i=1:num_neurons
            temp1=(d_log_lambda(:,i)'*lambda_pri(i)*dt*d_log_lambda(:,i));
            temp2=(spikes(i,k)-lambda_pri(i)*dt)*dd_log_lambda;
            temp3=temp3+temp1-temp2;
            temp4=temp4+d_log_lambda(:,i)'*(spikes(i,k)-lambda_pri(i)*dt);
        end
        W_post = 1/((W_pri)^(-1) + temp3);  % 卡尔曼增益
        x_est(m,k) = x_pri + W_post * temp4;  % 更新状态
    end
end
x_estimated=sum(x_est)/M;

% 结果展示
error_x = abs(x-x_estimated);
NMSE=norm(error_x)^2/norm(x)^2
figure
plot(1:N_t, x, 'k-' , 1:N_t, x_estimated, 'r--', 'LineWidth',2);
legend('True State', 'Estimated State');
title('State Estimation Using SPPE');
xlabel('Time Steps');
ylabel('State Value');
