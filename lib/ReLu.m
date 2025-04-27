function output = ReLu(input)
  % input: tensor with size [batch_size, height, width, feature_maps]
  % output: tensor after applying the ReLu Activation function with size [batch_size, height, width, feature_maps]
  
  % Input Size 
  [batch_size, height, width, feature_maps] = size(input);

  % Initialize ReLu Output
  output = zeros(batch_size, out_height, out_width, feature_maps);

  % ReLu Activation Loop
  for N = 1:batch_size
    for C = 1:feature_maps
      for W = 1:width
        for H = 1:height
          if (input(N,H,W,C) > 0)
            output(N,H,W,C) = input(N,H,W,C);
          else
            output(N,H,W,C) = 0;
          end
        end
      end
    end
  end

end
