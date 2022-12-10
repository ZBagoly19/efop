%% teszt adatok
X = Pos_X(4000);
Y = Pos_Y(4000);
Ori = Ori_Z(4000);
WA = WheAng_avg(4000);
V = Vel(4000);
Pos_X_rear = Pos_X_rearaxle(4000);
Pos_Y_rear = Pos_Y_rearaxle(4000);

[front_x, front_y] = calculate_middle_of_front_axel(X, Y, Ori);
wheel_base = get_wheelbase(front_x, front_y, Pos_X_rear, Pos_Y_rear);

kinematicModel = bicycleKinematics('WheelBase', wheel_base)
initialState = [Pos_X_rear Pos_Y_rear Ori];
tspan = 0 : 0.01 : 1;  % from now (0), with timestep 1 sec (1), 1 sec ahead (1)
inputs = [V WA]; %Turn left
[t,y] = ode45(@(t,y)derivative(kinematicModel, y, inputs),tspan,initialState);

figure(2)
plot(y(:,1),y(:,2))


%% functions
function [qx, qy] = calculate_middle_of_front_axel(ox, oy, angle)
    px = ox + 1.45; % 1.45: az auto kozepe es az elso tengely tavolsaga
    py = oy;

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy);
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy);
end

function wheel_base = get_wheelbase(fx, fy, rx, ry)
    x = fx - rx;
    y = fy - ry;

    wheel_base = sqrt(x*x + y*y)
end
