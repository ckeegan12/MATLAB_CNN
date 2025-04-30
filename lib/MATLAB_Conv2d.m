function output = Conv2d(input)
    % input: 4D tensor with size [height, width, channels, batch_size]
    % output: 4D tensor after applying the convolution with size [batch_size, height, width, channels]

    % Parameters
    [height, width, channels, batch_size] = size(input);  % Get input size
    feature_maps = 16;                                    % Number of feature maps
    kernel_size = 3;                                      % 3x3 Kernel 
    stride = 1;                                           % Stride
    padding = 1;                                          % Padding

    % Initialize Filters
    filters = randn(kernel_size, kernel_size, channels, feature_maps);  % 3x3xchannels filter for the 16 feature maps and channels
    

    % Apply Padding Around Input
    padded_input = padarray(input, [padding padding], 0, 'both');       % Pad with zeros around input

    % Output Size Calculation
    out_height = (height + 2 * padding - kernel_size) / stride + 1;     % Ouput height and convolution
    out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    % Initialize Bias
    bias = randn(1, 1, feature_maps);                                   % Bias for the 16 filters after convolution

    % Initialize Convolution Output
    output = zeros(out_height, out_width, feature_maps);
    % Convolution
    for N = 1:batch_size                   % Batch loop
        for K = 1:feature_maps             % Output feature maps loop
            for H = 1:out_height           % Output height loop
                for W = 1:out_width        % Output width loop
                    % Get input patch
                    h_start = (H-1)*stride + 1;
                    h_end = h_start + kernel_size - 1;
                    w_start = (W-1)*stride + 1;
                    w_end = w_start + kernel_size - 1;
                    
                    % Extract patch
                    patch = padded_input(h_start:h_end, w_start:w_end, :, N);
                    
                    % Compute convolution
                    conv_result = sum(patch .* filters(:,:,:,K), 'all');
                    
                    % Add bias
                    output(H, W, K, N) = conv_result + bias(1, 1, K);
                end
            end
        end
    end
end