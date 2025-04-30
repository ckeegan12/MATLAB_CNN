function output = Hidden_Layers(input)
  % Input: 2D matrix [batch_size, features]
  % Output: 2D matrix [batch_size, output_features]
  
  % Parameters
  [batch_size, input_features] = size(input);
  hidden_units = 128;                               % Number of neurons in hidden layers
  output_features = 10;                             % 10 classes for CIFAR-10
  num_layers = 3;
  
  % Initialize weights and biases
  weights = cell(1, num_layers);
  biases = cell(1, num_layers);
  
  % Layer 1
  weights{1} = randn(input_features, hidden_units) * 0.01;
  biases{1} = zeros(1, hidden_units);
  
  % Layer 2
  weights{2} = randn(hidden_units, hidden_units) * 0.01;
  biases{2} = zeros(1, hidden_units);
  
  % Layer 3
  weights{3} = randn(hidden_units, output_features) * 0.01;
  biases{3} = zeros(1, output_features);
  
  % Forward pass
  layer_output = input;
  
  for layer = 1:num_layers-1
      % Linear transformation
      layer_output = layer_output * weights{layer} + biases{layer};
      
      % ReLU activation
      layer_output = max(0, layer_output);
  end
  
  % Final layer
  output = layer_output * weights{3} + biases{3};
  
  % Softmax
  output = softmax(output, 2);
end