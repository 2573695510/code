%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
%% 导入数据
normalData = xlsread('正常风速数据.xlsx');
upData = xlsread('急剧上升风速数据.xlsx');
downData = xlsread('急剧下降风速数据.xlsx');
testData = readtable('测试集.xlsx'); % 转为 table 格式
win=xlsread('测试集.xlsx','F2:F97');
f_=5;
N1=length(normalData)-1;
N2=length(upData)-1;
N2=length(downData)-1;
%% 数据划分
P_train1 = normalData(1:N1, 1:5)';
T_train1 = normalData(1:N1, 6)';

P_train2 = upData(1:N2, 1:5)';
T_train2 = upData(1:N2, 6)';

P_train3 = downData(1:N3, 1:5)';
T_train3 = downData(1:N3, 6)';

P_test = testData{:, 1:5}';
T_test = testData{:, 6}';

%% 数据归一化
[p_train1, ps_input1] = mapminmax(P_train1, 0, 1);
[p_train2, ps_input2] = mapminmax(P_train2, 0, 1);
[p_train3, ps_input3] = mapminmax(P_train3, 0, 1);
p_test = mapminmax('apply', P_test, ps_input1);

[t_train1, ps_output1] = mapminmax(T_train1, 0, 1);
[t_train2, ps_output2] = mapminmax(T_train2, 0, 1);
[t_train3, ps_output3] = mapminmax(T_train3, 0, 1);
t_test = mapminmax('apply', T_test, ps_output1);

%%  格式转换 Transformer是seq2seq，因此输出不需要换成元胞
M = size(P_train1, 2);
N = size(P_test, 2);
for i = 1 : M 
    vp_train1{i, 1} = p_train1(:, i);
   
end

for i = 1 : M 
    vp_train2{i, 1} = p_train2(:, i);
   
end

for i = 1 : M 
    vp_train3{i, 1} = p_train3(:, i);
   
end
%% 输出
vt_train1 = t_train1';
vt_train2 = t_train2';
vt_train3 = t_train3';
vt_test = t_test';
%% 测试集
for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    % vt_test{i, 1} = t_test(:, i);
end



%% TCN_Transformer模型
clear lgraph
numChannels = f_;
maxPosition = 96;%原128 编码长度，一般是24小时*一小时数据条数
numHeads = 8; %原4 注意头数量 数据具有较多的特征间依赖关系，可能需要增加这个值
numKeyChannels = numHeads*16;%原16 如果数据特征复杂（如风电场时间序列数据存在多样性），可以尝试增加每头的通道数到 32

numFilters = 64; %卷积层的过滤器，原32 试将增加到 64 或 128，特别是当数据集较大或关系较为复杂时。如果你的数据集较小，也可以保持为 32，避免过拟合。
filterSize = 1; %原5 短期变化急剧使用减少，长依赖关系增加
dropoutFactor = 0.3; %如果发现过拟合，可以增加到 0.2 或 0.3；如果数据集较小，过拟合的风险较低，也可以保持较低的 dropout。
numBlocks = 4;%原2 TCN卷积块数量 可以增加模型的深度，更好地捕捉时间序列的依赖性。如果增加后出现过拟合，可以再调整回去。
layer = sequenceInputLayer(f_,Normalization="rescale-symmetric",Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor) 
        % spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor) 
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end


layers = [ 
    % sequenceInputLayer(numChannels,Name="input")
    positionEmbeddingLayer(numFilters,maxPosition,Name="pos-emb");%位置编码输入需要和TCN的输出对应
    additionLayer(2, Name="add")
    selfAttentionLayer(numHeads,numKeyChannels,'AttentionMask','causal')
    selfAttentionLayer(numHeads,numKeyChannels)
    indexing1dLayer("last")
    fullyConnectedLayer(1)
    regressionLayer];
lgraph = addLayers(lgraph,layers);


lgraph = connectLayers(lgraph,outputName,"pos-emb");
lgraph = connectLayers(lgraph,outputName,"add/in2");
plot(lgraph)
%%  参数设置 
options0 = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 200, ...                            % 最大训练次数
    'ValidationPatience', 20, ...                     %早停：验证损失连续20轮不下降则终止
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.001, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropPeriod', 50, ...        % 经过50次训练后 学习率为 0.001 * 0.5
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'L2Regularization', 0.0001, ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线


%% 网络训练
tic
net1 = trainNetwork(vp_train1,vt_train1,lgraph,options0);
net2 = trainNetwork(vp_train2,vt_train2,lgraph,options0);
net3 = trainNetwork(vp_train3,vt_train3,lgraph,options0);
toc
%% 测试集预测
windSpeedThreshold = 3; % 风速变化阈值
predictions = [];
isInTrend = false;      % 标记是否在持续趋势中
currentTrend = '';      % 当前趋势：'up' 或 'down'

for i = 1:height(testData)
    if ~isInTrend
        % 判断风速变化是否超过阈值
        if testData.WindSpeedChange1h(i) <= -windSpeedThreshold
            isInTrend = true;
            currentTrend = 'down';
        elseif testData.WindSpeedChange1h(i) >= windSpeedThreshold
            isInTrend = true;
            currentTrend = 'up';
        end
    end
    
    % 持续趋势内的预测
    if isInTrend
        if strcmp(currentTrend, 'down') && testData.WindSpeedChange1h(i) <= 0
            prediction = predict(net3, vp_test{i});
        elseif strcmp(currentTrend, 'up') && testData.WindSpeedChange1h(i) >= 0
            prediction = predict(net2, vp_test{i});
        else
            isInTrend = false;
            prediction = predict(net1, vp_test{i});
        end
    else
        % 正常风速数据预测
        prediction = predict(net1, vp_test{i});
    end
    
    predictions = [predictions; prediction];
end

%% 反归一化并保存结果
T_sim = mapminmax('reverse', predictions, ps_output1);
writematrix(T_sim, 'tcn预测结果.xlsx');
analyzeNetwork(net1);% 查看网络结构
%analyzeNetwork(net2);
%analyzeNetwork(net3);
%% 
MAE = mean(abs(win - T_sim));
RMSE = sqrt(sum((T_sim - win).^2)./96);
R = 1 - norm(win - T_sim)^2 / norm(win - mean(win))^2;
disp(['测试集数据的MAE为：', num2str(MAE)])
disp(['测试集数据的RMSE为：', num2str(RMSE)])
disp(['测试集数据的R^2为：', num2str(R)])