global loaded_net

%loaded_net = 'SZTAKI_weights_and_biases_acc666_cc0.mat';
%num_of_inputs = 49; num_of_outputs = 3;

%loaded_net = 'SZTAKI_weights_and_biases_acc333_cc0.mat';
%num_of_inputs = 49; num_of_outputs = 3;

%loaded_net = 'SZTAKI_weights_and_biases__csak_WheAng__speed50_cc0_acc333___.mat';
loaded_net = 'SZTAKI_weights_and_biases__csak_WheAng__speed60_cc0_acc333___.mat';
%loaded_net = 'SZTAKI_weights_and_biases__csak_WheAng__speed70_cc0_acc333___.mat';
num_of_inputs = 13; num_of_outputs = 1;

TS = 0.01;

XTrain_my2 = {10};
YTrain_my2 = {10};
XValidation_my2 = {2};
YValidation_my2 = {2};
for i = 1 : 850
    XTrain_my2{i, 1} = rand(num_of_inputs, 1);
    YTrain_my2{i, 1} = rand(num_of_outputs, 1);
end
for j = 1 : 27
    XValidation_my2{j, 1} = rand(num_of_inputs, 1);
    YValidation_my2{j, 1} = rand(num_of_outputs, 1);
end

numFeatures = size(XTrain_my2{1},1);
numResponses = size(YTrain_my2{1},1);

net = [
    sequenceInputLayer(numFeatures)
    fullyConnectedLayer(64, 'Name', 'fc_1', ...
                        'WeightsInitializer', @wiFC1, ...
                        'BiasInitializer', @biFC1)
    reluLayer()
    fullyConnectedLayer(64, 'Name', 'fc_2', ...
                        'WeightsInitializer', @wiFC2, ...
                        'BiasInitializer', @biFC2)
    reluLayer()
    fullyConnectedLayer(64, 'Name', 'fc_3', ...
                        'WeightsInitializer', @wiFC3, ...
                        'BiasInitializer', @biFC3)
    reluLayer()
    fullyConnectedLayer(64, 'Name', 'fc_4', ...
                        'WeightsInitializer', @wiFC4, ...
                        'BiasInitializer', @biFC4)
    reluLayer()
    
    fullyConnectedLayer(numResponses, 'Name', 'fc_last', ...
                        'WeightsInitializer', @wiFC_last, ...
                        'BiasInitializer', @biFC_last)
    regressionLayer()
 ]

options = trainingOptions('adam', ...
    'MaxEpochs',2, ...
    'MiniBatchSize', 850, ...
    'InitialLearnRate', 0.00000001, ...
    'ValidationData', {XValidation_my2, YValidation_my2}, ...
    'ValidationFrequency', 25, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'once', ...
    'Plots', 'none',...
    'Verbose', false);

global trained_net;
[trained_net, trained_net_info] = trainNetwork(XTrain_my2, YTrain_my2, net, options);

open generic_ipg_nn;

function weights = wiFC1(~)

global loaded_net
load(loaded_net, 'fc1w');
weights = transpose(extractdata(fc1w));

end

function weights = wiFC2(~)

global loaded_net
load(loaded_net, 'fc2w');
weights = transpose(extractdata(fc2w));

end

function weights = wiFC3(~)

global loaded_net
load(loaded_net, 'fc3w');
weights = transpose(extractdata(fc3w));

end

function weights = wiFC4(~)

global loaded_net
load(loaded_net, 'fc4w');
weights = transpose(extractdata(fc4w));

end

function weights = wiFC_last(~)

global loaded_net
load(loaded_net, 'fc_last_w');
weights = transpose(extractdata(fc_last_w));

end

function bias = biFC1(~)

global loaded_net
load(loaded_net, 'fc1b');
bias = extractdata(fc1b);

end

function bias = biFC2(~)

global loaded_net
load(loaded_net, 'fc2b');
bias = extractdata(fc2b);

end

function bias = biFC3(~)

global loaded_net
load(loaded_net, 'fc3b');
bias = extractdata(fc3b);

end

function bias = biFC4(~)

global loaded_net
load(loaded_net, 'fc4b');
bias = extractdata(fc4b);

end

function bias = biFC_last(~)

global loaded_net
load(loaded_net, 'fc_last_b');
bias = extractdata(fc_last_b);

end