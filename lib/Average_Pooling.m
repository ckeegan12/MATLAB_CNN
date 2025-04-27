function output = Average_Pooling(input)
  % input: tensor with size [batch_size, height, width, feature_maps]
  % output: tensor after applying the Max Pooling with size [batch_size, height, width, pooling_layers]

  % Parameters
  [batch_size, height, width, feature_maps] = size(input);  % Get input size
  Pool_layers = 16;                                     % Number of pooling layers
  Pool_size = 3;                                        % 3x3 Patch 
  stride = 3;                                           % Pooling stride

  % Output Parameters
  out_height = floor((height - Pool_size)/stride) + 1;
  out_width = floor((width - Pool_size)/stride) + 1;

  % Initialize Max Pooling Output
  output = zeros(batch_size, out_height, out_width, Pool_layers);

  % Perform convolution
  for N = 1:batch_size                                    % Batch loop
    for K = 1:feature_maps                                % Map loop
      for H = 1:out_height                                % Height loop
        for W = 1:out_width                               % Width  loop

          % Calculate input window coordinates
          h_start = (H-1)*stride(1) + 1;
          h_end = h_start + pool_size(1) - 1;
          w_start = (W-1)*stride(2) + 1;
          w_end = w_start + pool_size(2) - 1;
         
          % Extract window
          window = input(N, h_start:h_end, w_start:w_end, K);
         
          % Compute average
          output(N, H, W, K) = mean(window(:));

        end
      end
    end
  end

end
