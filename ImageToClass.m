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
%     imshow(s{i});
%     pause
% end

%% reorganise

la = repmat(1:8,1,10);
for i = 1:80
    k      = la(i);
    im{i}  = B{k};
    lab(i) = k;
    dat{1,i} = double(B{k}(:,:));
    dat3(:,:,1,i) =double(B{k}(:,:));
    label{1,i} = k;
    y          = cellstr(num2str(k));
    cata(i)    = categorical(y);
    cata2{i}   = categorical(y);
end
%%
DD= [[dat]; cata2];

    %% CNN

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(12,25)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

% lgraph = layerGraph(layers);
% plot(lgraph)

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'MaxEpochs',100, ...
    'Plots','training-progress');
net = trainNetwork(dat3,la',layers,options);

YPred = round(double(predict(net,dat3)));
YTest = la';
accuracy = sum(YPred == YTest)/numel(YTest)
analyzeNetwork(net)