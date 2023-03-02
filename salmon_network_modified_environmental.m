clear

%% read data from file
filename = '../../data/pied_piper/dungeness_2005-2020_combine_ma_msd.csv';
salmon_populations=readtable(filename);
%% select species to focus on (we chose 4, can add more)
%species_per_hour=[table2array(salmon_populations(1:147,6)), table2array(salmon_populations(1:147,27:28))]';%time-series data for 2005
%species_per_hour=[table2array(salmon_populations(148:329,6)), table2array(salmon_populations(148:329,27:28))]';%time-series data for 2006
species_per_hour=[table2array(salmon_populations(330:505,6)), table2array(salmon_populations(330:505,27:28)), table2array(salmon_populations(330:505,30))]';%time-series data for 2007

%We chose
%chinook0_wild_perhour, flow, temp, doy

%% Building Reservoir Computer model of Time-series

%% Random seed for reproducibility
rng(1,'twister') %can change it to rng(shuffle) for an ensemble of results with random reservoirs

%% Construction of the reservoir, specification of reservoir parameters
resparams=struct();
num_inputs = size(species_per_hour,1);
resparams.radius = 0.9; % spectral radius of reservoir
resparams.degree = 3; % average connection degree of reservoir
approx_res_size = 3000; % reservoir size (#nodes)
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = 0.1; % input weight scaling
resparams.train_length = size(species_per_hour,2)-5; % number of time points used to train 
resparams.num_inputs = num_inputs; 
resparams.beta = 10^(-4); %regularization parameter
resparams.leakage=1; %leakage rate
%% Training the reservoir 
[~, w_out, A, win,r] = train_reservoir(resparams, species_per_hour(:, 1:resparams.train_length),species_per_hour(:, 2:resparams.train_length+1));%Training
%This builds a trained reservoir that does one-step prediction of input
%time-series, i.e., it fits species_per_hour(:, 2:resparams.train_length+1) to species_per_hour(:, 1:resparams.train_length)

% w_out = reservoir-to-prediction coupling matrix
% win = input-to-reservoir coupling matrix
% A = Reservoir connectivity matrix
% r= Reservoir state vector at different times

%% Plot to check fitting to the data
figure
species_names=["chinook 0 wild","flow", "temp", "doy"];
species_per_hour_fitted=w_out*r;%output from the reservoir computer
for ispecies=1:size(species_per_hour,1)
    subplot(1,4,ispecies)
    plot(species_per_hour(ispecies,2:resparams.train_length+1),'-')
    hold on
    plot(species_per_hour_fitted(ispecies,1:resparams.train_length),'--')
    hold off
    title(species_names(ispecies))
    xlabel('Time step')
    ylabel('Flow per hour')
    legend({'Data','Fitted'},'location','northwest')
    set(gca,'FontSize',15)
end

%% Inferring the connection strength J, please refer to our paper for the equations used
J =zeros(size(win,2),size(win,2),resparams.train_length);
%3D array to store the N*N*T Jacobian elements (N=#species, T=#training data points)
%J(i,j,t) gives interaction strength from species j to species i at time-step t

B=A+win*w_out;
for it=1:resparams.train_length %time loop for calculation of J

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
%pairs=["chinook0 wild to hatchery","chinook0 hatchery to wild","coho1 hatchery to chinook0 wild","coho1 wild to chinook0 hatchery"];
%pairs=["coho1 wild to hatchery","coho1 hatchery to wild"];
pairs=["flow to chinook0 wild","temperature to chinook0 wild", "doy to chinook0 wild"];
p=zeros(3,size(J,3));
p(1,:)=J(1,2,:);%flow to chinook1 wild
p(2,:)=J(1,3,:);%temp to chinook0 wild
p(3,:)=J(1,4,:);%doy to chinook0 wild
%p(1,:)=J(1,2,:);%coho1 wild to coho1 hatchery
%p(2,:)=J(2,1,:);%coho1 hatchery to coho1 wild
%p(3,:)=J(2,3,:);%coho1 hatchery to chinook0 wild
%p(4,:)=J(1,4,:);%coho1 wild to chinook0 hatchery

for iline=1:3
subplot(1,3,iline)
plot(p(iline,:),'-','LineWidth',2)
xlabel('Time step')
ylabel('Interaction Strength')
set(gca,'FontSize',15)
title(pairs(iline))
end
sgtitle('Time-dependent interaction strengths')





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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w_out = train(params, states, data)
beta = params.beta;
idenmat = beta*speye(params.N);
w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%