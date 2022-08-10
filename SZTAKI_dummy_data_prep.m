XTrain_my2 = {850};
YTrain_my2 = {850};
XValidation_my2 = {27};
YValidation_my2 = {27};
for i = 1 : 850
    XTrain_my2{i, 1} = rand(43, 1);
    YTrain_my2{i, 1} = rand(1, 1);
end
for j = 1 : 27
    XValidation_my2{j, 1} = rand(43, 1);
    YValidation_my2{j, 1} = rand(1, 1);
end