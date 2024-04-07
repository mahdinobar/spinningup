%% Create Square Trajectory
% Create a square trajectory and examine the relationship between waypoint
% constraints, sample rate, and the generated trajectory.

clear all
clc
close all
%%

waypoints = [0.33086, -0.4753 ,  0.57432; ... % Initial position
             0.43086, -0.07530,  0.17432;...
             0.43086, 0.07530,  0.17432;...
             0.43086, 0.17530,  0.17432];    % Final position
waypoints_GroundSpeed = [0 ... % Initial position
             0.05...
             0.05...
             0.05];    % Final position


% toa = 0:80/240:80*1/240*(7-1); % time of arrival
freq=240;
dt=1/freq;
% orientation = quaternion([0,0,0; ...
%                           45,0,0; ...
%                           135,0,0; ...
%                           225,0,0; ...
%                           0,0,0], ...
%                           "eulerd","ZYX","frame");
% trajectory = waypointTrajectory(waypoints, ...
%     TimeOfArrival=toa, ...
%     Orientation=orientation, ...
%     SampleRate=1);
% trajectory = waypointTrajectory(waypoints, ...
%     TimeOfArrival=toa, ...
%     SampleRate=240);
trajectory = waypointTrajectory(waypoints, ...
    GroundSpeed=waypoints_GroundSpeed,...
    JerkLimit=0.5, ...
    SampleRate=freq);
%%
figure(1)
hold on
plot3(waypoints(1,1),waypoints(1,2),waypoints(1,3),"r*")
plot3(waypoints(end,1),waypoints(end,2),waypoints(end,3),"g*")
title("Position")
% axis([-1,2,-1,2])
axis square
xlabel("X")
ylabel("Y")
grid on

% orientationLog = zeros(toa(end)*trajectory.SampleRate,1,"quaternion");
count = 1;
view(3)
currentPosition_log=[];
currentAcc_log=[];
currentVel_log=[];
while ~isDone(trajectory)
   % [currentPosition,orientationLog(count)] = trajectory();
   [currentPosition,orientation,currentVel,currentAcc,angularVelocity] = trajectory();

   plot3(currentPosition(1),currentPosition(2),currentPosition(3),"ko")
   currentPosition_log=[currentPosition_log;currentPosition];
   currentAcc_log=[currentAcc_log;currentAcc];
   currentVel_log=[currentVel_log;currentVel];
   pause(trajectory.SamplesPerFrame/trajectory.SampleRate)
   count = count + 1;
end
hold off
%%
figure(2)
plot(currentPosition_log)
legend({"x","y","z"})
title("Pos")
%%
figure(3)
plot(currentAcc_log)
legend({"x","y","z"})
title("Acc")
%%
figure(4)
% plot(gradient(currentAcc_log,dt,dt,dt))
hold on
plot(gradient(currentAcc_log(:,1),dt),"r")
plot(gradient(currentAcc_log(:,2),dt),"g")
plot(gradient(currentAcc_log(:,3),dt),"b")
title("Jerk")
legend({"x","y","z"})
%%
figure(5)
plot(currentVel_log)
legend({"x","y","z"})
title("Vel")
%%
save("/home/mahdi/ETHZ/codes/rl_reach/code/logs/data_tmp.mat")
