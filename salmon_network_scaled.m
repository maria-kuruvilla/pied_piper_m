% Goal - To read new data file with all variables
% Remove rows with any NA
% Then run salmon network code for lag 0, lag 1, lag 2
% Divide interaction strength by (max of variable being influenced/ max of influencing variable) 
clear

%% read data from file
filename = '../../data/pied_piper/dungeness_2005-2014_subset2.csv';
salmon_data=readtable(filename,'PreserveVariableNames',true);

%% select varaibles to focus on
rc_data=[table2array(salmon_data(:,3:15)),table2array(salmon_data(:,18:19))]';%time-series data

names=["chinook0 hatchery"; "chinook0 wild"; "chinook1 hatchery"; "chinook1 wild"; "coho hatchery"; "coho wild"; "steel hatchery"; "steel wild"; "pink hatchery"; "pink wild"; "chum wild"; "flow"; "temp"; "atu solstice"; "photoperiod"]';

%% Building Reservoir Computer model of Time-series

%% Random seed for reproducibility
rng(1,'twister') %can change it to rng(shuffle) for an ensemble of results with random reservoirs

%% Construction of the reservoir, specification of reservoir parameters
resparams=struct();
num_inputs = size(rc_data,1);
resparams.radius = 0.9; % spectral radius of reservoir
resparams.degree = 3; % average connection degree of reservoir
approx_res_size = 3000; % reservoir size (#nodes)
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = 0.1; % input weight scaling
resparams.train_length = size(rc_data,2)-5; % number of time points used to train
resparams.num_inputs = num_inputs; 
resparams.beta = 10^(-4); %regularization parameter
resparams.leakage=1; %leakage rate


%% Training the reservoir 
[~, w_out, A, win,r] = train_reservoir(resparams, rc_data(:, 1:resparams.train_length),rc_data(:, 1:resparams.train_length));%Training
%This builds a trained reservoir that does one-step prediction of input
%time-series, i.e., it fits species_per_hour(:, 2:resparams.train_length+1) to species_per_hour(:, 1:resparams.train_length)
%1:resparams.train_length - predicting for the same time step
% w_out = reservoir-to-prediction coupling matrix
% win = input-to-reservoir coupling matrix
% A = Reservoir connectivity matrix
% r= Reservoir state vector at different times

%% Plot to check fitting to the data
figure
%species_names=["chinook0 hatchery","chinook0 wild","coho1 hatchery","coho1 wild"];
data_fitted=w_out*r;%output from the reservoir computer
for i=1:size(rc_data,1)
    subplot(3,5,i)
    plot(rc_data(i,1:resparams.train_length),'-')%can be 2:res + 1 if predicting for next time step
    hold on
    plot(data_fitted(i,1:resparams.train_length),'--')
    hold off
    title(names(i))
    xlabel('Time step')
    ylabel('Data')
    legend({'Data','Fitted'},'location','northwest')
    set(gca,'FontSize',15)
end
%% Inferring the connection strength J, please refer to our paper for the equations used
J =zeros(size(win,2),size(win,2),resparams.train_length);
%3D array to store the N*N*T Jacobian elements (N=#species, T=#training data points)
%J(i,j,t) gives interaction strength from species j to species i at time-step t

B=A+win*w_out;
for it=10:resparams.train_length %time loop for calculation of J %10 because we want to ignore the initial predictions

%track progress if you wish
%     if mod(it,100)==0
%        disp(it)
%     end
    
    x=r(:,it);
    A2=B*x;
    mat1=zeros(size(w_out));
    for i1=1:resparams.N
        mat1(:,i1)=w_out(:,i1)*(sech(A2(i1)))^2;
    end
    
h=(mat1*(win+A*pinv(w_out)));

J(:,:,it)=h(:,:);
end

%% Plot interactions over time
figure
%we pick some specific connection strengths to visualize
pairs=["chinook0 hatchery","chinook0 wild","chinook0 wild to chinook0 hatchery","chinook0 hatchery to chinook0 wild"];
p=zeros(4,size(J,3));
p(1,:) = rc_data(1,1:resparams.train_length);
p(2,:) = rc_data(2,1:resparams.train_length);
p(3,:)=J(1,2,:)*(max(rc_data(2,:))/max(rc_data(1,:)));%chinook0 wild (2) to chinook0 hatchery (1)
p(4,:)=J(2,1,:)*(max(rc_data(1,:))/max(rc_data(2,:)));%chinook0 hatchery to chinook0 wild


for iline=1:4
subplot(2,2,iline)
plot(p(iline,:),'-','LineWidth',2)
xlabel('Time step')
ylabel('Interaction Strength')
set(gca,'FontSize',15)
title(pairs(iline))
end
sgtitle('Time-dependent interaction strengths')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, wout, A, win,states] = train_reservoir(resparams, data1,data2)

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);
q = resparams.N/resparams.num_inputs;
win = zeros(resparams.N, resparams.num_inputs);
for i=1:resparams.num_inputs
    ip = resparams.sigma*(-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end
states = reservoir_layer(A, win, data1, resparams);
wout = train(resparams, states, data2(:,1:resparams.train_length));
x = states(:,end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = generate_reservoir(size, radius, degree)
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
beta = params.beta;
idenmat = beta*speye(params.N);
w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
