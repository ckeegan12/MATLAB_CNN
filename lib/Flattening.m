function output = Flattening(input)
  % input: 4D tensor with size [batch_size, height, width, feature_maps]
  % output: 2D matrix of flattened values (batch_size x (height*width*feature_maps))

  % Get input size
  [batch_size, height, width, feature_maps] = size(input);

  % Preallocate output
  output = zeros(batch_size, height * width * feature_maps);

  % Flatten each sample in the batch
  for B = 1:batch_size

      % Extract the B-th sample: 3D tensor (height x width x feature_maps)
      sample = squeeze(input(B, :, :, :));  
      
      % Flatten the 3D tensor into a 1D vector
      output(B, :) = sample(:);  
      
  end
end