clc;clear all;close all;
% LQG Elbow-shoulder angular position, velocity , motor commands for 8
% reach direction
load('SimuData2.mat');
for i = 1:size(Dat2.In,1)
xTra{i,1} = [Dat2.In(i,1:51); Dat2.In(i,52:102);Dat2.In(i,103:153);Dat2.In(i,154:204)];
tTra{i,1} = [Dat2.Out(i,1:4)'];
end

%% visualise  decoded data
for i= 1:160
    % plot path in joint space
    p1 =  Dat2.In(i,1:51);
    p2 =  Dat2.In(i,52:102);
    v1 =  Dat2.In(i,103:153);
    v2 =  Dat2.In(i,154:204);
    subplot(131);plot(p1,p2,'.k'); title('Pos');hold on
    subplot(132);plot(v1,v2,'.k'); title('Vel');
    hold on;
%     pause
    subplot(133); plot(Dat2.Out(i,3),Dat2.Out(i,4),'o'); hold on;
end
% axis image;
subplot(132);xlabel('shoulder (rad)');
ylabel('elbow (rad)');
%%
figure
plot(xTra{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(xTra{1},1);
legend("Feature " + string(1:numFeatures))

%% 
for  i = 1:size(Dat2.In,1)
    xTra2{i,1} = Dat2.Out(i,:)';
    yTra2{i,1} = Dat2.In(i,:)';
    xTes2{i,1} = Dat2.In(i,:)';
end

%% lstm
numFeatures = 4;
numResponses = 304;
numHiddenUnits = 500;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(500,'Name','bilstm1') %lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(xTra2,yTra2,layers,options);
%%
analyzeNetwork(net)
%% predict
for i = 1:8
YPred(:,i) = predict(net,xTra2{i,1});
plot(YPred(1:51,i),YPred(52:102,i));hold on;
end

netAngToTrajLSTM  = net;
save netAngToTrajLSTM