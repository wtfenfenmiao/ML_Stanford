function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);            %功能就是X把那个平均的减掉，用bsxfun要比循环或者矩阵操作高效的多得多得多

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);      %和上面同理。bsxfun的意思说不太清，但是上网搜一下，或者定义一个[1 1 1;2 4 8;3 9 27]的例子作为X，执行一下代码，一看就明白

% ============================================================

end
