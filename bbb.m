function out = bbb(input)

global trained_net
result = SeriesNetwork/predict(trained_net, input, 'MiniBatchSize', 1);

% [out, state] = netFcn(input, params);
out = result;

end