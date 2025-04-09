function output = Conv2d(input)
    % input: 4D tensor with size [height, width, channels, batch_size]
    % output: 4D tensor after applying the convolution with size [height, width, 16, batch_size]

    % Parameters
    [height, width, channels, batch_size] = size(input);  % Get input size
    out_channels = 16;   % Number of output channels
    kernel_size = 3;     % Kernel size
    stride = 1;          % Stride
    padding = 1;         % Padding

    % Define the filters (kernels) randomly
    filters = randn(kernel_size, kernel_size, channels, num_filters);  % 3x3 filter, 3 input channels, 16 output channels

    % Apply padding to the input (padding of 1 means 1 pixel around all sides)
    padded_input = padarray(input, [padding padding], 0, 'both');  % Pad with zeros

    % Output size calculation
    out_height = (height + 2 * padding - kernel_size) / stride + 1;
    out_width = (width + 2 * padding - kernel_size) / stride + 1;

    % Initialize the output
    output = zeros(out_height, out_width, num_filters, batch_size);

    % Perform convolution
    for n = 1:N                  % Batch loop
        for c_out = 1:C_out      % Output channels
            for h = 1:outH       % Height loop
                for w = 1:outW   % Width  loop
                    h_start = (h-1)*stride(1) + 1;
                    h_end = h_start + kH - 1;
                    w_start = (w-1)*stride(2) + 1;
                    w_end = w_start + kW - 1;
                    
                    % Extract input patch
                    patch = input(h_start:h_end, w_start:w_end, :, n);
                    
                    % Element-wise multiplication and sum
                    conv_sum = sum(patch .* weights(:,:,:,c_out), 'all');
                    
                    output(h, w, c_out, n) = conv_sum;
                end
            end
        end
    end
end
