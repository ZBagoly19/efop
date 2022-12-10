WheAng_avg = zeros(length(WheAng_FR), 1);

for i = 1: length(WheAng_FR)
    WheAng_avg(i, 1) = (WheAng_FR(i, 1) + WheAng_FL(i, 1)) / 2;
end