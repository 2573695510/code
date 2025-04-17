function [error_pos] = iso_forest(res)
%%  超参数设置
rng("default")                         % 固定随机种子
contaminationFraction = single(0.05);  % 设置异常比例 

%%  孤立森林进行检测
for i = 1:size(res,2)
    data = [res(:,i) res(:,end)];
    [forest, tf_forest(:,i), s_forest] = iforest(data, ContaminationFraction = contaminationFraction);
end
%% 异常值位置
for i =1:size(res,2)
    error_pos{i,1} = find(tf_forest(:,i)==1);
end
%%  可视化降维
% T = tsne(res, Standardize = true);

%%  绘制可视化结果
% figure
% gscatter(res_new(:, 1),res_new(:, end), tf_forest(:,1), "br", [], 15, "off")
% legend("正常值", "离群值")
% title("孤立森林")
% set(gcf,'color','w')
% 
% 
% res(error_pos,:)=[];
end