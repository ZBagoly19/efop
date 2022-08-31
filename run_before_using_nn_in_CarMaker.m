TS = 0.01;

XTrain_my2 = {10};
YTrain_my2 = {10};
XValidation_my2 = {2};
YValidation_my2 = {2};
for i = 1 : 850
    XTrain_my2{i, 1} = rand(49, 1);
    YTrain_my2{i, 1} = rand(3, 1);
end
for j = 1 : 27
    XValidation_my2{j, 1} = rand(49, 1);
    YValidation_my2{j, 1} = rand(3, 1);
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

function weights = wiFC1(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc1w');
weights = transpose(extractdata(fc1w));

end

function weights = wiFC2(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc2w');
weights = transpose(extractdata(fc2w));

end

function weights = wiFC3(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc3w');
weights = transpose(extractdata(fc3w));

end

function weights = wiFC4(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc4w');
weights = transpose(extractdata(fc4w));

end

function weights = wiFC_last(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc_last_w');
weights = transpose(extractdata(fc_last_w));

end

function bias = biFC1(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc1b');
bias = extractdata(fc1b);

end

function bias = biFC2(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc2b');
bias = extractdata(fc2b);

end

function bias = biFC3(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc3b');
bias = extractdata(fc3b);

end

function bias = biFC4(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc4b');
bias = extractdata(fc4b);

end

function bias = biFC_last(~)

load('SZTAKI_weights_and_biases_longlong08.mat', 'fc_last_b');
bias = extractdata(fc_last_b);

end