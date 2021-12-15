%% sys id 
clear 


% here we are going to estimate two system parameters (mass and offset)
% from data (we are going to create this data) 

%% data creation 
mass = 4;
offset = 0.1;
p_true = [mass;offset]; % these are the truth values 

% simulate dynamics with true values
dt = 0.5;
N = 10;
X = zeros(4,N);
U = zeros(2,N-1);

x0 = [3;2;0.1;-0.3];
X(:,1) = x0;
for i = 1:N-1
    U(:,i) = [sin(i);1.3*cos(i)];
    X(:,i+1) = rk4(X(:,i),U(:,i),p_true,dt);
end

% plot trajectories
figure
hold on 
title('positions')
plot(X(1:2,:)')
legend('px','py')
hold off 


figure
hold on 
title('velocities')
plot(X(3:4,:)')
legend('vx','vy')
hold off 

%% check cost function 

p1 = p_true*0.5;

% this should be 0 with the true p value
J = cost_function(p_true,X,U)

% but nonzero with an incorrect p value 
J = cost_function(p1,X,U)


%% now we solve to p to see if we can converge on the true p 
options = optimset('Display','iter','PlotFcns',@optimplotfval);
p_guess = randn(2,1);
p_solve = fminsearch(@(p)cost_function(p,X,U),p_guess,options)


%% supporting functions

function J = cost_function(p,X,U)
% Xtilde is experimental state history 
% Utilde is experimental control history 
% p contains system parameters (mass and offset) that we are trying to
% solve for. 

J = 0;
N = size(X,2);
dt = 0.5;
for i = 1:N-1
    % add the norm squared of the error between the two steps, as predicted
    % given our dynamics model with current p values 
    J = J + norm(X(:,i+1) - rk4(X(:,i),U(:,i),p,dt))^2;
end
end



function xdot = dynamics(x,u,p)

% position 
r = x(1:2);

% velocity 
v = x(3:4);

% unpack p (this is where I put the system parameters i'm solving for)
mass = p(1);   % mass of the particle
offset = p(2); % offset 

% kinematics
rdot = v;

% dynamics
vdot = u/mass + offset;

xdot = [rdot;vdot];
end

function xkp1 = rk4(x,u,p,dt)
k1 = dt*dynamics(x,u,p);
k2 = dt*dynamics(x + k1/2,u,p);
k3 = dt*dynamics(x + k2/2,u,p);
k4 = dt*dynamics(x + k3,u,p);
xkp1 = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4);
end