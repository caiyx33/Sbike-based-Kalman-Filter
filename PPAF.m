%% 初始化
clear;
clc;
%close all;

% 参数设定
T = 60;                   % 总时间，秒
dt = 0.01;
t = 0:dt:T-dt;            % 时间向量
N_t=T/dt;
rho=1;
alpha = 1;                   % 背景发射率
beta = 3*ones(1,N_t);                 % 调制参数

% 生成频率调制的三角波
f0 = 0.01;                 % 初始频率，Hz
f1 = 0.1;                   % 结束频率，Hz
k = (f1 - f0) / T;        % 频率斜率
f = f0 + k * t;           % 瞬时频率
omega = cumsum(f) * 2*pi*dt; % 积分以获得相位
x = 1 * sawtooth(omega, 0.5);  % 生成三角波
num_neurons=1;
% 添加高斯噪声
n_v = 2.5e-5;
noise = sqrt(n_v) * randn(size(x));
x_noisy = x + noise;

% 计算神经元发射率
lambda = exp(alpha + beta .* x_noisy);  % 发射率函数

% 生成神经元尖峰
spikes = rand(1, N_t) < lambda * dt;

%% 绘制仿真神经信号与偏好方向
figure;
subplot(2,1,1);
plot((1:N_t) * dt, x);
title('One-dimensional kinematics (Random Walk)');
xlabel('Time (s)');
ylabel('Position');
subplot(2,1,2);
plot((1:N_t) * dt, lambda);
title('Change of Lambda for Neuron 1');
xlabel('Time (s)');
ylabel('Lambda');

figure;
plot((1:N_t) * dt, lambda);
title(['Lambda for Neuron ']);
xlabel('Time (s)');
ylabel('Lambda');
figure;
hold on;
for i = 1:num_neurons
    spk_times = find(spikes(i, :));
    for j = 1:length(spk_times)
        plot([1, 1] * spk_times(j) * dt, [i-0.4, i+0.4], 'k');
    end
end
ylim([0.5, num_neurons + 0.5]);
title('尖峰火花栅格图');
xlabel('时间 (s)');
ylabel('神经元索引');
x_estimated=zeros(num_neurons,1);
beta_estimated=zeros(num_neurons,1);
x_estimated(1)=-1;
beta_estimated(1)=3;

% 参数初始化
xx=x(2:N_t);
XX=x(1:N_t-1);
F = xx*XX'/(XX*XX');
Q=0.001;%用拟合的运动模型计算过程噪声协方差
M=10;
x_est = zeros(M, N_t); % 初始化状态估计
x_est(:,1)=-1;
W_post = 1; % 初始化方差估计，假设为1
% 运行滤波算法
for m=1:M
    for k = 2:N_t
        % 预测步
        x_pri = F * x_est(m,k-1);
        W_pri = F * W_post * F' + Q;
    
        % 计算lambda
        lambda_pri = rho * exp(alpha + beta(1, k) * x_pri); % 计算尖峰率
        log_lambda = log(rho) + alpha + beta(1, k) * x_pri;
        d_log_lambda = beta(1, k);  % log(lambda) 对 theta 的一阶导数
        dd_log_lambda = 0;          % log(lambda) 对 theta 的二阶导数为 0
    
        % 更新方差和状态
        temp1=(d_log_lambda'*lambda_pri*dt*d_log_lambda);
        temp2=(spikes(1,k)-lambda_pri*dt)*dd_log_lambda;
        W_post = 1/((W_pri)^(-1) + (temp1-temp2));  % 卡尔曼增益
        x_est(m,k) = x_pri + W_post * d_log_lambda' * (spikes(1, k) -lambda_pri*dt);  % 更新状态
    end
end
x_estimated=sum(x_est)/M;
% 结果展示
error_x = abs(x-x_estimated);
NMSE=norm(x_estimated-x)/norm(x)
figure
plot(1:N_t, x, 'k-' , 1:N_t, x_estimated, 'r-', 'LineWidth',2);
legend('True State', 'Estimated State');
title('State Estimation Using SPPE');
xlabel('Time Steps');
ylabel('State Value');