function output = SoftMax(input)
  % input: Array output from neural network
  % output: probability array from SoftMax function

  % Sample size
  samples = size(input);

  % Initialize output and parameters
  output = zeros(samples);
  dem = 0;
  num = 0;

  % For loop for Denominator of SoftMax
  for N = 1:samples      
      
    % Denominator for SoftMax
    dem = dem + exp(N);

  end
  
  % Loop for SoftMax
  for S = 1:samples

    % Numerator for SoftMax
    num = exp(S);

    % SoftMax fucntion
    output(S) =  num / dem; 

  end
end
