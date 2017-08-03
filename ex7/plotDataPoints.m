function plotDataPoints(X, idx, K)
%PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
%index assignments in idx have the same color
%   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
%   with the same index assignments in idx have the same color

% Create palette
palette = hsv(K + 1);
colors = palette(idx, :);
%��һ��(��һ��)�ܻ��ǣ�Ҫѧϰ
%b=[1;2;3;3;2;1;1;2;3]    c=[52;45;43;23;11;22;33;44;55]   d=c(b,:),Ȼ����Եõ�,
%d =[52;45;43;43;45;52;52;45;43]

% Plot the data
scatter(X(:,1), X(:,2), 15, colors);

end
