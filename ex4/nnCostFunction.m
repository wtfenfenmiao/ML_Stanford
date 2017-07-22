function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%这里改了特别多次才对，都是根据gradient checking的数不多来的，这个梯度检验真的很有用！！！！
%backpropagation只是为了算grad的，这里只是J和grad，外面还要用梯度下降fmincg等等的算法来训练参数
%注意理解，这里的运算，都是在一种theta值下，算出来J和grad，算J就简单粗暴的直接往前一步步算，m没关系，在m上不用循环；
%算grad就要1到m循环，每一步的bigdelta值都不一样（在1到m的过程中更新），但是theta值一直没变过啊，这么循环算bigdelta只是为了算出在当前theta下的grad
%步骤如下：
%首先把所有的都放在一起（m个例子一起算就行）算当前theta下的输出h(theta)(这里的a3)
%之后用此输出算J
%之后用backpropagation算grad，这里要用for循环，逐个例子算（把m拆开）
%算的过程套公式就行，不过要格外注意一点，涉及到x*theta的时候，x要在前面补一个1，而不是theta，theta在x=1的那个分支的值也要一点点算的（详情见pdf的神经网络图，能看到是x补了1，而不是theta）
%在之前的算法里，都是类似于theta0+theta1*x,然后为了方便想成theta0*1,这也是补一个x=1,而不是让theta0=1,千万别混了

%算J
z2=[ones(size(X,1),1) X]*Theta1';
a2=sigmoid(z2);               %m*25   delta和对应的a行列数相同
z3=[ones(size(X,1),1) a2]*Theta2';
a3=sigmoid(z3);         
ytemp=zeros(size(y,1),size(Theta2,1));
for i=1:size(y,1)
  ytemp(i,y(i))=1;
end
%J=sum(sum(-(log(a3).*ytemp)-(1-ytemp).*(log(1-a3))))/m;       %Feedforward and Cost Function
J=sum(sum(-(log(a3).*ytemp)-(1-ytemp).*(log(1-a3))))/m+(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);


%backpropagation算grad,套公式
bigDelta1=zeros(size(Theta1));
bigDelta2=zeros(size(Theta2));
for i=1:m                  %算delta3和delta2感觉感觉感觉（别试了，按照这个吧，清晰明了，否则又蒙了）是可以放在外面的，因为这个每个都是只跟本次m相关的，就是一个训练例子有一个delta2和delta3，要循环的和前一次相关的只是bigdelta
  delta3=a3(i,:)-ytemp(i,:);         %1*10
  delta2=((delta3*Theta2).*[1 sigmoidGradient(z2(i,:))])(:,2:end);         %1*25,这里的想法就是x和a，delta这种可以补1，如果多了就把1去了，不要动theta的维度，只动补1那里的维度，补1是为了方便的
  bigDelta1=bigDelta1+delta2'*[1 X(i,:)];                  %25*401
  bigDelta2=bigDelta2+delta3'*[1 a2(i,:)];             %10*26
end
  

Theta1_grad=bigDelta1/m+lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=bigDelta2/m+lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
