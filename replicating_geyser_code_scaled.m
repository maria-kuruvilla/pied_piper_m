% Goal - to replicate geyser code to try different delays and to calculate 
% interaction strength for each month

clear

%compare=zeros(18,12,10,10);

%1 month=43200 min;
delays=[0,1,2,3];%daily sampling - 6 day delay
dl_labels=["0 day","1 day", "2 day", "3 day"];
delays=floor(delays);

%% read data from file
filename = '../../data/pied_piper/dungeness_2005-2014_subset2.csv';
salmon_data=readtable(filename,'PreserveVariableNames',true);

%% select varaibles to focus on
x=[table2array(salmon_data(:,3:15)),table2array(salmon_data(:,18:19))]';%time-series data

names=["chinook0 hatchery"; "chinook0 wild"; "chinook1 hatchery"; "chinook1 wild"; "coho hatchery"; "coho wild"; "steel hatchery"; "steel wild"; "pink hatchery"; "pink wild"; "chum wild"; "flow"; "temp"; "atu solstice"; "photoperiod"]';

compare = zeros(1,4,15,15);
for imonth=1:1
for idel=1:4
   disp([imonth,idel])

idelay = delays(1,idel);
data=x(:,1:1500);

diff_data=x(:,1+idelay:1500+idelay);

%% Data for Reservoir training
measurements1 = data(:,1:end);% + z;
measurements2 = diff_data(:, 1:end);

resparams=struct();
%% Train reservoir
num_inputs = size(measurements1,1);
resparams.radius = 0.9; % spectral radius of reservoir
resparams.degree = 3; % average connection degree of reservoir
approx_res_size = 3000; % reservoir size (#nodes)
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = 0.1; % input weight scaling
resparams.train_length = size(data,2)-5; % number of time points used to train - I am changing to size(fish, 1) - 50
resparams.num_inputs = num_inputs; 
resparams.beta = 10^(-4); %regularization parameter
resparams.leakage=1; %leakage rate
resparams.predict_length = 100; % number of predictions after training
resparams.predict_length_max=resparams.predict_length;
resparams.bias=0;%bias


%% Resrovir dynamics
[xx, w_out, A, win,r] = train_reservoir(resparams, measurements1,measurements2);%Train and save w_out matrix for future use

%% geyser code

%% Jacobian Estimation
av_length=floor(resparams.train_length*0.8);

%% If predicting x(t+dt)from x(t)
conn1=zeros(size(win,2));
B=A+win*w_out;
for it=resparams.train_length-av_length:resparams.train_length
    
    if mod(it,1000)==0
    disp(it)
    end
    
    xx=r(:,it);
    A2=B*xx;
        
    mat1=zeros(size(w_out));

    for i1=1:resparams.N
        mat1(:,i1)=w_out(:,i1)*(sech(A2(i1)))^2;
    end
    
           conn1=conn1+(mat1*(win+A*pinv(w_out))); %abs(mat1*(win+A*pinv(w_out)))
%           conn1=conn1+(mat1*(win+A*pinv(w_out)));

end
conn1=conn1/(av_length);

compare(imonth,idel,:,:)=conn1(:,:);
end
end

figure
for idel1=1:4
disp(idel1)
subplot(2,2,idel1)
conn1=zeros(11);
conn1(:,:)=compare(1,idel1,1:11,1:11);
imagesc(conn1-diag(diag(conn1)))
pbaspect([1 1 1])
xticks(1:1:11)
yticks(1:1:11)
xticklabels(names)
yticklabels(names)
xtickangle(45)
ytickangle(45)
title(dl_labels(1,idel1))
colorbar
set(gca,'FontSize',10)
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, wout, A, win,states] = train_reservoir(resparams, data1,data2)

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);
q = resparams.N/resparams.num_inputs;
win = zeros(resparams.N, resparams.num_inputs);
for i=1:resparams.num_inputs
     rng(i+50,'twister')
    ip = resparams.sigma*(-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end
states = reservoir_layer(A, win, data1, resparams);
wout = train(resparams, states, data2(:,1:resparams.train_length));
x = states(:,end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = generate_reservoir(size, radius, degree)
  rng(1,'twister');
sparsity = degree/size;
while 1
A = sprand(size, size, sparsity);
e = max(abs(eigs(A)));

if (isnan(e)==0)%Avoid NaN in the largest eigenvalue, in case convergence issues arise
    break;
end

end
A = (A./e).*radius;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function states = reservoir_layer(A, win, input, resparams)

states = zeros(resparams.N, resparams.train_length);
for i = 1:resparams.train_length-1
    states(:,i+1) = (1-resparams.leakage)*states(:,i)+resparams.leakage*tanh(A*states(:,i) + win*input(:,i));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w_out = train(params, states, data)

%Take some time points out of training if you want
% T=size(states,2);
% states(:,floor(T/2)-20:floor(T/2)+20)=[];
% data(:,floor(T/2)-20:floor(T/2)+20)=[];



beta = params.beta;
rng(2,'twister');
idenmat = beta*speye(params.N);
% states(2:2:params.N,:) = states(2:2:params.N,:).^2;
w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=residual(x,data1,data2,rho)
measured_vars = 1:1:size(data1,1);
num_measured = length(measured_vars);
measurements1 = data1(measured_vars, :);% + z;
measurements2 = data2(measured_vars, :);% + z;

resparams=struct();
%train reservoir
[num_inputs,~] = size(measurements1);
resparams.radius = rho; % spectral radius, around 0.9
resparams.degree = x(1); % connection degree, around 3
approx_res_size = 3000; % reservoir size
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = x(2); % input weight scaling, around 0.1
resparams.leakage=1;
resparams.train_length = 30000; % number of points used to train
resparams.num_inputs = num_inputs; 
resparams.predict_length = 1000; % number of predictions after training
resparams.beta = 0.0001; %regularization parameter, near 0.0001

[~, H, ~, ~,r] = train_reservoir(resparams, measurements1,measurements2);
output1=H*r;
f=sum((output1(:,floor(resparams.train_length*0.1):resparams.train_length)-data2(:,floor(resparams.train_length*0.1):resparams.train_length)).^2,'all');
end
