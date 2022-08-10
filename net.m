% layers = [
%     imageInputLayer([1 43 1],   'Name', 'input', ...
%                                 'Normalization', 'none')
%     fullyConnectedLayer(64,     'Name','fc1', ...
%                                 'WeightsInitializer', @weightsInitFC1)
%     fullyConnectedLayer(64,     'Name','fc2',       @weightsInitFC2)
%     fullyConnectedLayer(64,     'Name','fc3',       @weightsInitFC3)
%     fullyConnectedLayer(64,     'Name','fc4',       @weightsInitFC4)
%     fullyConnectedLayer(1,      'Name','fc last',   @weightsInitFC_LAST)
% ]
layers = [
    imageInputLayer([1 43 1],   'Name', 'input', ...
                                'Normalization', 'none')
    fullyConnectedLayer(64,     'Name', 'fc1')
    fullyConnectedLayer(64,     'Name', 'fc2')
    fullyConnectedLayer(64,     'Name', 'fc3')
    fullyConnectedLayer(64,     'Name', 'fc4')
    fullyConnectedLayer(1,      'Name', 'fc last')
]

function weights = weightsInitFC1(sz)

x = 10;
y = 2;
result = randn(x, y);
weights = result;

end