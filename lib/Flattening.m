function output = Flattening(input)
  % input: 4D tensor [height, width, feature_maps, batch_size]
  % output: 2D matrix [batch_size, height*width*feature_maps]
  
  % Get input dimensions
  [height, width, feature_maps, batch_size] = size(input);
  
  % Reshape tensor
  output = reshape(permute(input, [4, 1, 2, 3]), batch_size, []);
  
  % Alternative version:
  % output = zeros(batch_size, height * width * feature_maps);
  % for B = 1:batch_size
  %     sample = input(:,:,:,B); 
  %     output(B,:) = sample(:)';
  % end

end