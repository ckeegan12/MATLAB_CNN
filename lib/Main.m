% CNN main fucntion

% Cifar10 Data Set download
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder,'cifar-10-matlab.tar.gz');

dataFolder = fullfile(downloadFolder,'cifar-10-batches-mat');
if ~exist(dataFolder,'dir')
    fprintf("Downloading CIFAR-10 dataset (175 MB)... ");
    websave(filename,url);
    untar(filename,downloadFolder);
    fprintf("Done.\n")
end

% Cifar10 training and testing sets 
[X_Train,Y_Train,X_Test,Y_Test] = loadCIFARData(downloadFolder);

%%%%%%%%% Feature Learning %%%%%%%%%

% Convolution + ReLu Layer 1
Conv_1 = MATLAB_Conv2d(X_Train);
Out_1 = ReLu(Conv_1);

% Average Pooling Layer 1
Pooling_out_1 = Average_Pooling(Out_1);

% Convolution + ReLu Layer 2
Conv_2 = MATLAB_Conv2d(Pooling_out_1);
Out_2 = ReLu(Conv_2);

% Average Pooling Layer 2
Pooling_out_2 = Average_Pooling(Out_2);

%%%%% Feature Learing Complete %%%%%

%%%%%%% Classification Layer %%%%%%%

Flatten_array = Flattening(Pooling_out_2);

output = SoftMax();