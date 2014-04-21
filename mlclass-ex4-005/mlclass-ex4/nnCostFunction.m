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
%fprintf('size of Theta1_grad .\n');
%size(Theta1_grad )
Theta2_grad = zeros(size(Theta2));
%fprintf('size of Theta2_grad .\n');
%size(Theta2_grad )

%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
% need to recode y, which is of 5000 * 1 with value 1 to 10, to 5000 * 10
y_recoded = zeros(size(y, 1), size(unique(y), 2));
labels = unique(y)';
for c= labels,
	y_recoded(:, c) = (y==c);
end;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% compute hypothesis function by following ex3.pdf Fig 2
% Add ones to the X data matrix for the bias node,
% which doesn't depend on any of the input value X
a1 = [ones(m, 1) X];
	%fprintf('size of a1.\n');
	%size(a1)
a2 = [ones(m, 1) sigmoid(a1*Theta1')];
	%fprintf('size of a2.\n');
	%size(a2)
h = sigmoid(a2*Theta2');
% h is now a matrix of 5000 * 10, w/ each column containing the prob of the given label

% for the formula, need to sum it twice
% inner sum(,2) to sum all the cost for across all columns (each label)
% outer sum() for summing the cost for all observations
J = sum(sum(-y_recoded.*log(h) - (1-y_recoded).*log(1-h), 2))/m ...
	+ ( sum(sum(Theta1(:, 2:end).^2, 2))+ sum(sum(Theta2(:, 2:end).^2, 2))) * lambda/(2*m);

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
for t=1:m,
	delta3 = h(t, :) - y_recoded(t, :);
	%fprintf('size of delta3.\n');
	%size(delta3);
	Theta2_grad	= Theta2_grad + delta3'*a2(t, :);

	delta2 = (delta3 * Theta2);
	delta2 = delta2(2:end) .* sigmoidGradient(a1(t, :)*Theta1');
	Theta1_grad	= Theta1_grad + delta2'*a1(t, :);
	%fprintf('size of delta2.\n');
	%size(delta2);
end

Theta2_grad	= Theta2_grad/m;
Theta1_grad	= Theta1_grad/m;
%
% WHAT DO WE WANT?
% WE WANT dCost/dTheta2 and dCost/dTheta1 for gradient descent
%
%
% dCost/dTheta2 = dCost/dzL3 * dzL3/dTheta2
%
% 1st term
% http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
% Cost	= 0.5*( h - y )^2
%		= 0.5*( g(zL3) - y )^2	where g = 1/(1+e^-z) with dg/dz = g*(1-g)
% this is the cost for one observation.  will need to divide by m for multiple
% dCost/dzL3	= dCost/dg * dg/dzL3
% 				= ( g - y ) * g*(1-g)
%dCost_dzL3	= (h - y_recoded) .* h .* (ones(size(h,1), size(h,2)) - h)
%fprintf('size of dCost_dzL3.\n');
%size(dCost_dzL3)

% 2nd term
% zL3	= Theta2 * aL2
% dzL3/dTheta2 = aL2
%
% So, 
% dCost/dTheta2	= dCost/dzL3 * dzL3/dTheta2
% 				= (g-y)*g*(1-g) * aL2	where g = g(zL3) = h = aL3
%
%
% dCost/dTheta1 is a bit more tricky
% dCost/dTheta1 = dCost/dzL2 * dzL2/dTheta1
%
% 1st term
% dCost/dzL2	= dCost/dzL3 * dzL3/dzL2
% zL3	= Theta2 * aL2
%		= Theta2 * g(zL2)
% dzL3/dzL2	= dzL3/dg(zL2) * dg(zL2)/dzL2
%			= Theta2 * g*(1-g)	where g = g(zL2)
% dCost/dzL2	= dCost/dzL3 * dzL3/dzL2
% 				= dCost/dzL3 * Theta2 * g*(1-g)
% 				= [dCost/dzL3 * Theta2] * g'(zL2)
% note that in [dCost/dzL3 * Theta2], dCost/dzL3 is the "error term" from next layer and we back propagate it by the means of Theta2 with the weighted average
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




Theta2_grad	= Theta2_grad + [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]*lambda/m;
Theta1_grad	= Theta1_grad + [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]*lambda/m;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
