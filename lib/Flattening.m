function output = Flattening(input)
  % input: 4D tensor with size [batch_size, height, width, Pooling_Layers]
  % output: 1D array of Average_Pooling values after applying the Flattening function

  % Parameters
  [batch_size, height, width, feature_maps] = size(input);  % Get input size

  output = zeros(batch_size, (height * width * feature_maps));

  for B = 1:batch_size
    for 

end
