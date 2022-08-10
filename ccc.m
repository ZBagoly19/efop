function [output] = ccc(input)
    output = predict(trained_net, input, 'MiniBatchSize', 1);
end