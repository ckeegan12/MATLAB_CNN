function output = Hidden_Layers(input,weights,biases)
  % Input: 2D matrix [batch_size, features]
  % Weights: 128 node weights for hidden layers
  % Biases: 128 node bias for hidden layers 
  % Output: 2D matrix [batch_size, output_features]
  
  % Parameters
  [batch_size, input_features] = size(input);
  num_layers = length(weights);  

  % Forward pass
  layer_output = input;
  
  for layer = 1:num_layers
      % Linear transformation
      layer_output = layer_output * weights{layer} + biases{layer};
      
      % ReLU for hidden layers
      if layer < num_layers
          layer_output = max(0, layer_output);
      end
  end
  
  % Softmax output
  output = softmax(layer_output, 2);

end