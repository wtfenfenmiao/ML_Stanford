function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);            %���ܾ���X���Ǹ�ƽ���ļ�������bsxfunҪ��ѭ�����߾��������Ч�Ķ�ö�ö�

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);      %������ͬ��bsxfun����˼˵��̫�壬����������һ�£����߶���һ��[1 1 1;2 4 8;3 9 27]��������ΪX��ִ��һ�´��룬һ��������

% ============================================================

end
