function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% My code: deciding which is the best value of C and sigma to minimize cost
C_list = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_list = [0.01;0.03;0.1;0.3;1;3;10;30];
cost_best = 0;
sigma_best = 0;
C_best = 0;
for i = 1:8
 for j = 1:8
  C_temp = C_list(i);
  sigma_temp = sigma_list(j);
  model_temp = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
 % this gives us the model that best fits X and y
 % now to see which one fits best for Xval and yval
  predic_val = svmPredict(model_temp,Xval);
  cost_temp = mean(double(predic_val~=yval));
  if(i==1 && j==1)
   cost_best = cost_temp;
   C_best = C_temp;
   sigma_best = sigma_temp;
  elseif(cost_temp < cost_best)
   cost_best = cost_temp;
   C_best = C_temp;
   sigma_best = sigma_temp;
  end
 end
end
 % now we have the best value of C and sigma

 C = C_best;
 sigma = sigma_best;
 fprintf("Best value of C: %f and sigma %f\n",C,sigma);






% =========================================================================

end
