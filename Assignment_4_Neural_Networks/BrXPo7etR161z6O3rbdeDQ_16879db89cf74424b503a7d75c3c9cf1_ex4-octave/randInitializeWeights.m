function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
%   of a layer with L_in incoming connections and L_out outgoing
%   connections.
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%


Y = zeros(input_layer_size,num_labels);
for c = 1:num_labels
  pos = find(y==c);
  Y(pos,c)=1;
endfor

a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2_size = size(a2, 1);
a2 = [ones(a2_size,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;
for c = 1:m
  h = a3(c,:)';
  yk = Y(c, :)';
  mat = -yk.*log(h) - (1 - yk).*log(1-h);
  J += sum(mat);
endfor
J/=m;
regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regularator;



for c = 1:m
  a1 = [1;X(c,:)'];
  z2 = Theta1*a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  yy = Y(c,:)';
  delta_3 = a3 - yy;
  delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
  delta_2 = delta_2(2:end);
  Theta1_grad = Theta1_grad + delta_2 * a1';
  Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];







% =========================================================================
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
