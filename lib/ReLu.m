function output = ReLu(input)
  % input: 4D tensor with size [batch_size, height, width, channels]
  % output: 4D tensor after applying the convolution with size [batch_size, height, width, channels]
  
  % Input Size 
  [batch_size, height, width, channels] = size(input);

  % Initialize ReLu Output
  output = zeros(batch_size, out_height, out_width, feature_maps);

  % ReLu Activation Loop
  for N = 1:batch_size
    for H = 1:height
      for W = 1:width
        for C = 1:channels
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