function output = ReLu(input)
  % input: 4D tensor [height, width, feature_maps, batch_size]
  % output: 4D tensor of same size after ReLU activation

  % Initializing output
  [height, width, feature_maps, batch_size] = size(input);
  output = zeros(size(input));
  
  % ReLu Loop: element-wise activation function
  for N = 1:batch_size
      for C = 1:feature_maps
          for W = 1:width
              for H = 1:height
                  if input(H,W,C,N) > 0
                      output(H,W,C,N) = input(H,W,C,N);
                  else
                      output(H,W,C,N) = 0;
                  end
              end
          end
      end
  end
  
end