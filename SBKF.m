function X_estimated=SBKF(a,spikes,F,Q,dt,rho,alpha,beta)
    [num_neurons,N_t]=size(spikes);
    X_estimated=zeros(1,N_t); %保存每次SPPE状态估计的MC结果
    X_estimated(1)=a;
    W_post=0.01;
    for t = 2:N_t
        % 预测步
        X_pri = F * X_estimated(:,t-1);
        W_pri = F * W_post * F' + Q;
        % 计算lambda
        lambda_pri = rho * exp(alpha + beta(:,t)' * X_pri); % 计算尖峰率
        log_lambda = log(rho) + alpha + beta(:,t)' * X_pri;
        d_log_lambda = beta(:,t)';  % log(lambda) 对 theta 的一阶导数
        dd_log_lambda = 0;          % log(lambda) 对 theta 的二阶导数为 0
        % 更新方差和状态
        temp3=0;temp4=0;
        for i=1:num_neurons
            temp1=(d_log_lambda(:,i)'*lambda_pri(i)*dt*d_log_lambda(:,i));
            temp2=(spikes(i,t)-lambda_pri(i)*dt)*dd_log_lambda;
            temp3=temp3+temp1-temp2;
            temp4=temp4+d_log_lambda(:,i)'*(spikes(i,t)-lambda_pri(i)*dt);
        end
        W_post = 1/((W_pri)^(-1) + temp3);  % 卡尔曼增益
        X_estimated(t) = X_pri + W_post * temp4;  % 更新状态
    end
end