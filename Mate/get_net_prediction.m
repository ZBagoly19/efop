function out = get_net_prediction(input)

global trained_net

pred = predict(trained_net, input, 'MiniBatchSize', 1);
out = transpose(double(pred));

end
