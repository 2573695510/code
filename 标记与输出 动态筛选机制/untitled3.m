% 读取数据
data = readtable('清洗.xlsx'); 

% 设置滑动窗口和阈值
windowSize = 1; 
windSpeedThreshold = 3; % 风速突变阈值，例如一样本数内下降或上升超过3 m/s

% 提取风速列
windSpeed = data.ws;
windSpeedChange = NaN(height(data), 1); % 初始化风速变化列

% 计算1小时风速变化
for i = windowSize + 1 : height(data)
    windSpeedChange(i) = windSpeed(i) - windSpeed(i - windowSize); % 计算风速变化
end

% 将风速变化结果添加到数据表中
data.WindSpeedChange1h = windSpeedChange;

% 初始化标记列（标记急剧变化的开始位置和方向）
dramaticWindChangeUpData = []; % 初始化空数组用于保存急剧上升的数据
dramaticWindChangeDownData = []; % 初始化空数组用于保存急剧下降的数据
isInTrend = false; % 标记是否在持续变化的趋势中
trend = ''; % 当前的风速变化趋势，'down'表示下降，'up'表示上升

% 筛选出风速变化超过阈值的突变数据并检查持续变化
for i = 2 : height(data)
    if ~isInTrend
        % 判断风速变化是否超过阈值
        if data.WindSpeedChange1h(i) <= -windSpeedThreshold % 风速急剧下降
            trend = 'down'; % 设置为下降趋势
            isInTrend = true; % 进入持续下降的状态
            dramaticWindChangeDownData = [dramaticWindChangeDownData; data(i-1, :)]; % 保存突变前一条数据
            dramaticWindChangeDownData = [dramaticWindChangeDownData; data(i, :)];   % 保存突变的数据
        elseif data.WindSpeedChange1h(i) >= windSpeedThreshold % 风速急剧上升
            trend = 'up'; % 设置为上升趋势
            isInTrend = true; % 进入持续上升的状态
            dramaticWindChangeUpData = [dramaticWindChangeUpData; data(i-1, :)]; % 保存突变前一条数据
            dramaticWindChangeUpData = [dramaticWindChangeUpData; data(i, :)];   % 保存突变的数据
        end
    else
        % 根据当前趋势继续记录风速数据
        if strcmp(trend, 'down') && data.ws(i) < data.ws(i-1) % 如果是下降趋势并且风速继续下降
            dramaticWindChangeDownData = [dramaticWindChangeDownData; data(i, :)]; % 保存继续下降的数据
        elseif strcmp(trend, 'up') && data.ws(i) > data.ws(i-1) % 如果是上升趋势并且风速继续上升
            dramaticWindChangeUpData = [dramaticWindChangeUpData; data(i, :)]; % 保存继续上升的数据
        else
            % 风速变化趋势发生反转，结束记录
            isInTrend = false; 
        end
    end
end

% 将筛选出的上升和下降数据分别保存为Excel文件
writetable(data, '总数据.xlsx');
writetable(dramaticWindChangeUpData, '急剧上升风速数据.xlsx');
writetable(dramaticWindChangeDownData, '急剧下降风速数据.xlsx');
