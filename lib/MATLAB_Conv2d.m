function output = Conv2d(input)
    % input: 4D tensor with size [batch_size, height, width, channels]
    % output: 4D tensor after applying the convolution with size [batch_size, height, width, channels]

    % Parameters
    [batch_size, height, width, channels] = size(input);  % Get input size
    feature_maps = 16;                                    % Number of feature maps
    kernel_size = 3;                                      % 3x3 Kernel 
    stride = 1;                                           % Stride
    padding = 1;                                          % Padding

    % Initialize Filters
    filters = randn(kernel_size, kernel_size, feature_maps);            % 3x3xchannels filter for the 16 feature maps
    

    % Apply Padding Around Input
    padded_input = padarray(input, [padding padding], 0, 'both');       % Pad with zeros around input

    % Output Size Calculation
    out_height = (height + 2 * padding - kernel_size) / stride + 1;     % Ouput height and convolution
    out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    % Initialize Bias
    bias = randn(out_height, out_width, feature_maps);                  % Bias for the 16 feature maps after convolution

    % Initialize Convolution Output
    output = zeros(batch_size, out_height, out_width, feature_maps);

    % Perform convolution
    for N = 1:batch_size                                % Batch loop
        for C = 1:channels                              % Channels loop
            for K = 1:feature_maps                      % Filter loop
                for H = 1:out_height                    % Height loop
                    for W = 1:out_width                 % Width  loop
                    
                        % Input Image elements
                        h_start = (H-1)*stride + 1;
                        h_end = h_start + kernel_size - 1;
                        w_start = (W-1)*stride + 1;
                        w_end = w_start + kernel_size - 1;

                        % Convolution Patch
                        Patch = padded_input(N, h_start:h_end, w_start:w_end, :);

                        % Convolution Of Kernel And Patch
                        output(N,H,W,K) = sum(Patch .* filters(:,:,K)) + bias(H,W,K);
                    end
                end
            end
        end
    end
end

