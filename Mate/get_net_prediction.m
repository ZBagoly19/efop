function out = bbb(input)

global trained_net

pred = predict(trained_net, input, 'MiniBatchSize', 1);
out = transpose(double(pred));

end