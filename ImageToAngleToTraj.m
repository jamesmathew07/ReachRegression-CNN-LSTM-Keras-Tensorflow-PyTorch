clc;clear all;close all;
% A test case with pretrained CNN, LSTM nets

%load test data, pretrained CNN, LSTM

load('dat3.mat')
load netIMtoAng.mat
load netAngToPos.mat
load netAngToTrajLSTM.mat

YPred = predict(netIMtoAng,dat3);
Out1  = netAngToPos(YPred');

%% visualise  decoded data
for i= 1:160
    % plot path in joint space
    p1 =  Out1(1:51,i);
    p2 =  Out1(52:102,i);
    v1 =  Out1(103:153,i);
    v2 =  Out1(154:204,i);
    subplot(131);plot(p1,p2,'.k'); title('Pos');hold on
    subplot(132);plot(v1,v2,'.k'); title('Vel');
    hold on;

end
% axis image;
subplot(132);xlabel('shoulder (rad)');
ylabel('elbow (rad)');


%% Test
t = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M7.tif','r');
S1   = read(t);
s{7} = rgb2gray(S1);
B{7} = imresize(s{7},[28 28]);
TestImage(:,:,1,1) = B{7}(:,:);

YPred = predict(netIMtoAng,TestImage(:,:,1,1));
Out2  = netAngToPos(YPred');

subplot(141); imshow(S1);title('Cartesian Work space')
subplot(142); plot(YPred(1,3),YPred(1,4),'or'); hold on;
plot(pi/2,pi/2,'or'); hold on;
xlim([0.8,2.4]);ylim([0.8,2.4]);axis square;title('Joint space');
xlabel('shoulder (rad)');
subplot(143);
p1 =  Out2(1:51,1);
p2 =  Out2(52:102,1);
v1 =  Out2(103:153,1);
v2 =  Out2(154:204,1);
plot(p1,p2,'.k'); title('Pos');hold on;
xlim([0.8,2.4]);ylim([0.8,2.4]);
xlabel('shoulder (rad)');
ylabel('elbow (rad)');
axis square
subplot(144);plot(v1,v2,'.k'); title('Vel');
xlabel('shoulder (rad/s)');
ylabel('elbow (rad/s)');
xlim([-2.5,2.5]);ylim([-2.5,2.5]);
axis square

%% Lstm
t    = Tiff('C:\Users\jmathew\Dropbox (INMACOSY)\James-UCL\LQG\iLQG\Cues\M2.tif','r');
S1   = read(t);
s{7} = rgb2gray(S1);
B{7} = imresize(s{7},[28 28]);
TestImage(:,:,1,1) = B{7}(:,:);

YPred = predict(netIMtoAng,TestImage(:,:,1,1));
Out2  = predict(netAngToTrajLSTM,YPred');

subplot(141); imshow(S1);title('Cartesian Work space')
subplot(142); plot(YPred(1,3),YPred(1,4),'or'); hold on;
plot(pi/2,pi/2,'or'); hold on;
xlim([0.6,2.4]);ylim([1.4,2.4]);
axis square;title('Joint space');
xlabel('shoulder (rad)');
subplot(143);
p1 =  Out2(1:51,1);
p2 =  Out2(52:102,1);
v1 =  Out2(103:153,1);
v2 =  Out2(154:204,1);
plot(p1,p2,'.k'); title('Pos');hold on;
xlim([0.6,2.4]);ylim([0.8,2.4]);
xlabel('shoulder (rad)');
ylabel('elbow (rad)');
axis square
subplot(144);plot(v1,v2,'.k'); title('Vel');
xlabel('shoulder (rad/s)');
ylabel('elbow (rad/s)');
xlim([-2.5,2.5]);ylim([-2.5,2.5]);
axis square
