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

a1 = X;
temp2 = [ones(1,m);X'];
%temp now has all the sample inputs as columns, with a 1 at the top (a(1))
z2 = Theta1*temp2;
%temp now has z(2)
a2 = sigmoid(z2);
c = size(a2,2);
temp3 = [ones(1,c);a2];
%temp now has a(2)
z3 = Theta2*temp3;
%temp now has z(3)
a3 = sigmoid(z3);
%temp now has a(3) or h(x) where each column, is that input's truth matrix for 1..10

num1 = [];
num2 = [];

for i = 1:num_labels
 num1 = [num1;i];
 num2 = [num2;i];
end

for i = 1:(m-1)
 num1 = [num1,num2];
end
%num1 now has 1...10 written in the m columns
temp_ans = y';
%temp now has all the answers in a row
truth_matrix = (num1==temp_ans);
%now truth matrix has those columns with 1's and 0's
J = -sum(sum(truth_matrix.*log(a3)+(1-truth_matrix).*log(1-a3)))/m;
%this is the unregularized cost
%to add the sum of squares of all theta's except the bias values
toAdd = sum(sum(Theta1.^2))-sum(Theta1(:,1).^2)+sum(sum(Theta2.^2))-sum(Theta2(:,1).^2);
toAdd = lambda*toAdd/(2*m);
J = J + toAdd;
%J now has the regularized cost
del3 = a3-truth_matrix;
del2 = Theta2'*del3;
del2 = del2([2:end],:);
del2 = del2.*sigmoidGradient(z2);
Theta2_grad = del3*(temp3')/m;
Theta1_grad = del2*(temp2')/m;
num_zeros1 = hidden_layer_size;
num_zeros2 = num_labels;
Theta1_grad = Theta1_grad + lambda/m*[zeros(num_zeros1,1),Theta1(:,[2:end])];
Theta2_grad = Theta2_grad + lambda/m*[zeros(num_zeros2,1),Theta2(:,[2:end])];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
