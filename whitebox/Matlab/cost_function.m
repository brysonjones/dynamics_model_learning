function J = cost_function(p,X,U)
% Xtilde is experimental state history 
% Utilde is experimental control history 
% p contains system parameters (mass and offset) that we are trying to
% solve for. 

J = 0;
N = size(X,2);
dt = 0.5;
for ii = 1:N-1
    % add the norm squared of the error between the two steps, as predicted
    % given our dynamics model with current p values 
    J = J + norm(X(:,ii+1) - rk4(X(:,ii),U(:,ii),p,dt))^2;
end
end