function output = myConv2d(input, weights, stride, padding)
    % Custom 2D convolution mimicking PyTorch's nn.Conv2d
    % Input: 4D tensor [H, W, C_in, N]
    % Weights: 4D tensor
    % Stride: [strideH, strideW]
    % Padding: 'same'
    
    % Convert PyTorch weight format [C_out, C_in, kH, kW] to MATLAB-friendly [kH, kW, C_in, C_out]
    weights = permute(weights, [3, 4, 2, 1]);  
    
    [H, W, C_in, N] = size(input);
    [kH, kW, ~, C_out] = size(weights);
    
    if strcmp(padding, 'same')
        padH = floor((kH - 1) / 2);
        padW = floor((kW - 1) / 2);
        input = padarray(input, [padH, padW], 0, 'both');
    else
        error('Only "same" padding is supported in this implementation.');
    end
    
    % Compute output dimensions
    outH = floor((H + 2*padH - kH) / stride(1)) + 1;
    outW = floor((W + 2*padW - kW) / stride(2)) + 1;
    
    output = zeros(outH, outW, C_out, N);
    
    % Perform convolution
    for n = 1:N                 % Batch loop
        for c_out = 1:C_out     % Output channels
            for h = 1:outH       % Height loop
                for w = 1:outW   % Width loop
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
