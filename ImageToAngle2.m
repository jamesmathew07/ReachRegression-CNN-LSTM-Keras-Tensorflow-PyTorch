clc;close all;clear all
% load images of reach targets
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M1.tif','r');
S1 = read(t);
s{1} = rgb2gray(S1);
B{1} = imresize(s{1},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M2.tif','r');
S1 = read(t);
s{2} = rgb2gray(S1);
B{2} = imresize(s{2},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M3.tif','r');
S1 = read(t);
s{3} = rgb2gray(S1);
B{3} = imresize(s{3},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M4.tif','r');
S1 = read(t);
s{4} = rgb2gray(S1);
B{4} = imresize(s{4},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M5.tif','r');
S1 = read(t);
s{5} = rgb2gray(S1);
B{5} = imresize(s{5},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M6.tif','r');
S1 = read(t);
s{6} = rgb2gray(S1);
B{6} = imresize(s{6},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M7.tif','r');
S1 = read(t);
s{7} = rgb2gray(S1);
B{7} = imresize(s{7},[28 28]);
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M8.tif','r');
S1 = read(t);
s{8} = rgb2gray(S1);
B{8} = imresize(s{8},[28 28]);

%% 
% for i = 1:8
%     imshow(B{i});
%     pause
% end

%% LQG Elbow-shoulder angular position, velocity , motor commands for 8
% reach direction
load('SimuData2.mat');
xTra = Dat2.In';
tTra = Dat2.Out';

%%
la = repmat(1:8,1,20);
for i = 1:160
    k             = la(i);
    dat3(:,:,1,i) = double(B{k}(:,:));
end

    %% CNN
layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(12,25)
    reluLayer
    fullyConnectedLayer(4)
    regressionLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'MaxEpochs',200, ...
    'Plots','training-progress');
net = trainNetwork(dat3,tTra',layers,options);
netIMtoAng = net;
YPred = predict(net,dat3);
YTest = tTra';
analyzeNetwork(net)

%% test
for i = 1:160
plot(YPred(i,1),YPred(i,1),'*r');hold on;plot(YPred(i,3),YPred(i,4),'*r');
plot(YTest(i,1),YTest(i,1),'ob');hold on;plot(YTest(i,3),YTest(i,4),'ob');
end

%%
save netIMtoAng
save('dat3.mat','dat3')