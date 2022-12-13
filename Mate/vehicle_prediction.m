%% prediction time
tspan_end = 2;

%% test data
load('meas_cornercut0_speed50_acc333_longlong_sens.mat')
chosen_point = 4000;
X = Pos_X(chosen_point);
Y = Pos_Y(chosen_point);
Ori = Ori_Z(chosen_point);
WA = WheAng_avg(chosen_point);
V = Vel(chosen_point);
Pos_X_rear = Pos_X_rearaxle(chosen_point);
Pos_Y_rear = Pos_Y_rearaxle(chosen_point);

[front_x, front_y] = rotate_and_translate(X, Y, Ori, 1.45);
wheel_base = get_wheelbase(front_x, front_y, Pos_X_rear, Pos_Y_rear)


%% prediction
% this uses Robotics System Toolbox
kinematicModel = bicycleKinematics('WheelBase', wheel_base)
initialState = [Pos_X_rear Pos_Y_rear Ori];
tspan = 0 : 0.01 : tspan_end;  % from now : with timestep 0.01 sec : 1 sec ahead
inputs = [V WA]; %Turn left
[t, y] = ode45(@(t, y)derivative(kinematicModel, y, inputs), tspan, initialState);
pred_length = size(y);
pred_rear_X = y(pred_length(1), 1);
pred_rear_Y = y(pred_length(1), 2);
pred_Ori = y(pred_length(1), 3);
[predicted_front_x, predicted_front_y] = rotate_and_translate(pred_rear_X, pred_rear_Y, pred_Ori, wheel_base);

% figure(1)
% axis equal
% hold on
% %plot(PathX, PathY)
% plot(PathX(chosen_point-100:chosen_point+300), PathY(chosen_point-100:chosen_point+300))
% plot(y(:,1),y(:,2))
% 
P0x = predicted_front_x
P0y = predicted_front_y

P1x = P0x + 1
P1y = P0y + tan(pred_Ori + pi/2)
% plot([P0x, P1x], [P0y, P1y])
% 
% legend('path', ...
%     'prediction of middle of rear axel', ...
%     'perpendicular line to orientation from middle of front axel')


%% intersection
perpendicular_line_x = [P0x, P1x]
perpendicular_line_y = [P0y, P1y]
% this uses Fast and Robust Curve Intersections
robust = 1;
[x0, y0, iout, jout] = intersections(PathX, PathY, perpendicular_line_x, perpendicular_line_y, robust)

% figure(2)
% axis equal
% hold on
% %plot(PathX, PathY)
% plot(PathX(chosen_point-100:chosen_point+300), PathY(chosen_point-100:chosen_point+300));
% plot(y(:,1),y(:,2));
% 
% P0x = predicted_front_x;
% P0y = predicted_front_y;
% 
% P1x = P0x + 1;
% P1y = P0y + tan(pred_Ori + pi/2);
% plot([P0x, P1x], [P0y, P1y]);
% 
% plot(x0, y0, 'o');
% 
% legend('path', ...
%     'prediction of middle of rear axel', ...
%     'perpendicular line to orientation from middle of front axel', ...
%     'intersection')


%% lateral distance
figure(3)
axis equal
hold on
%plot(PathX, PathY)
plot(PathX(chosen_point-100:chosen_point+300), PathY(chosen_point-100:chosen_point+300));
plot(y(:,1),y(:,2));

P0x = predicted_front_x;
P0y = predicted_front_y;

P1x = P0x + 1;
P1y = P0y + tan(pred_Ori + pi/2);
plot([P0x, P1x], [P0y, P1y]);

plot(x0, y0, 'o');
plot(predicted_front_x, predicted_front_y, 'o');

legend('path', ...
    'prediction of middle of rear axel', ...
    'perpendicular line to orientation from middle of front axel', ...
    'intersection', ...
    'middle of front axel')

LATERAL_ERROR = sqrt((predicted_front_x-x0)*(predicted_front_x-x0) + (predicted_front_y-y0)*(predicted_front_y-y0))


%% functions
function [qx, qy] = rotate_and_translate(ox, oy, angle, tra)
    px = ox + tra;
    py = oy;

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy);
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy);
end

function wheel_base = get_wheelbase(fx, fy, rx, ry)
    x = fx - rx;
    y = fy - ry;

    wheel_base = sqrt(x*x + y*y);
end
