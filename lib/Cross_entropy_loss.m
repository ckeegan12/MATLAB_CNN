function loss = Cross_entropy_loss(Y_Train,input)
  % input: SoftMax output from hidden_layers size: [batch_size,probability distributiom]
  % Y_Train: classes of the the X_Train
  % Loss: difference in class probability

  [batch_size, probability] = size(input);
  Y_Train_onehot = onehotencode(Y_Train, 2);

  % Cross-entropy loss parameters (avoid log(0) and input larger than 1)
  epsilon = 1e-12;
  input = max(input, epsilon);
  input = min(input, 1 - epsilon);
  
  % Compute cross-entropy loss
  loss = -sum(sum(Y_Train_onehot .* log(input))) / batch_size;
end