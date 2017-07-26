function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
%随着训练数据的增加，theta每当训练数据不一样的时候就会训练出一个不一样的值（所以theta的取值放在循环里）
%然后用训练的这些数据和Xval的总数据比（重要，下面算error的时候error_train是用的训练的数据数，但是Xval用总数，这个训练的数据数i和Xval是无关的，它影响theta，而且有几个训练数据Jtrain就用几个是对的，但是Xval和这个训练的数据数i无关！！！！）
%而且还把图给记混了，high bias是两个曲线差不多接上了（交叉验证集在上，训练集在下），而且是平的，所以训练数据再加也没用
%high variance是两个曲线中间差了一大块（交叉验证集在上，训练集在下），但是有接近的趋势，所以增加训练集的数量是有用的
for i=1:m
    theta = trainLinearReg(X(1:i,:), y(1:i,:), lambda);
    error_train(i)=sum((X(1:i,:)*theta-y(1:i,:)).^2)/(2*i);
    error_val(i)=sum((Xval*theta-yval).^2)/(2*size(Xval,1));
end




% -------------------------------------------------------------

% =========================================================================

end
