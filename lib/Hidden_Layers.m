function output = Hidden_Layers(input)
  % Input: array output from feature learning
  % Output: array for SoftMax and BackPropagation functions

  % Parameters
  rowVec = reshape(input, 1, []);        % Input changed to row vector
  size = length(rowVec);                 % Elements in input 
  layers = 3;                            % Number of hidden layers

  % Initialization
  weights = randn(size,1,layers);        % Initializing weights as column vector
  bias = randn(size,1,layers);           % Initializing biases as column vector
  layer1_output = zeros()

  %%%%%%% Layer 1 %%%%%%%

  % Forward Propagation
  for M = 1:size
    layer1_output(M) = rowVec(M) * weights(M,1,1) + bias(M,1,1);
  end

  % ReLu
  for N = 1:size
    if layer1_output(N) < 0 
      layer1_output(N) = 0;
    end
  end

  %%%%%%% Layer 2 %%%%%%%

  % Forward Propagation
  for B = 1:size
    layer2_output(B) = layer1_output(B) * weights(B,1,2) + bias(B,1,2);
  end

  % ReLu
  for V = 1:size
    if layer2_output(V) < 0 
      layer2_output(V) = 0;
    end
  end

  %%%%%%% Layer 3 %%%%%%%

  % Forward Propagation
  for C = 1:size
    layer3_output(C) = layer2_output(C) * weights(C,1,3) + bias(C,1,3);
  end

  % ReLu
  for X = 1:size
    if layer3_output(X) < 0 
      layer3_output(X) = 0;
    end
  end

end