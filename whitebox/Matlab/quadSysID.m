%% sys id 
% clear 


% here we are going to estimate two system parameters (mass and offset)
% from data (we are going to create this data) 

%% data creation 
kt = 0.005;
km = 0.0005;
J = diag([.1, .1, .2]);
p_true = [kt; km; J(:)]; % these are the truth values 

% simulate dynamics with true values
dt = 0.5;
N = numel(t);
X = zeros(13,N);
U = zeros(4,N-1);

pos0  = [0; 0; 10]; %starting airborne
quat0 = [1; 0; 0; 0]; %aligned with world
vel0  = [0; 0; 0]; %floating, slightly forward
w0    = [0; 0; 0];
x0 = [pos0; quat0; vel0; w0];

% t = dt * (1:N);
% X(:,1) = x0;
% baseVal = 375;
% for ii = 1:N-1
%     U(:,ii) = baseVal + 10*[cos(ii); sin(ii); -2*cos(ii); -2*sin(ii)];
%     X(:,ii+1) = rk4(X(:,ii),U(:,ii),p_true,dt);
% end

U = [mot1, mot2, mot3, mot4]';
X = [ posX, posY, posZ, quatW, quatX, quatY, quatZ, velX, velY, velZ, angVelX, angVelY, angVelZ ]';

% plot trajectories
figure
hold all 
title('positions')
plot(t, X(1,:))
plot(t, X(2,:))
plot(t, X(3,:))
legend('x','y','z')
hold off 


figure
hold all 
title('velocities')
plot(t, X(8,:))
plot(t, X(9,:))
plot(t, X(10,:))
legend('x','y','z')
hold off 

figure; hold all
euls = zeros(3, N);
for ii = 1:N
    euls(:, ii) = quat2eul(X(4:7,ii)');
end
plot(t, euls(1, :))
plot(t, euls(2, :))
plot(t, euls(3, :))
xlabel('time (sec)')
ylabel('Angles')
legend('Z', 'Y', 'X')

%% check cost function 

p1 = p_true*0.5;
% this should be 0 with the true p value
Jtmp = cost_function(p_true,X,U)

% but nonzero with an incorrect p value 
Jtmp = cost_function(p1,X,U)


%% now we solve to p to see if we can converge on the true p 
options = optimset('Display','iter','PlotFcns',@optimplotfval);
ktGuess = 0.1;
kmGuess = 0.1;
JGuess = eye(3);
p_guess = [ktGuess; kmGuess; JGuess(:)];
p_solve = fminsearch(@(p)cost_function(p,X,U),p_guess,options)

kt_calc = p_solve(1)
km_calc = p_solve(2)
J_calc = reshape(p_solve(3:end), 3, 3)
