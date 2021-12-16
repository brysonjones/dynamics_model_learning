%Next State MSE evaluation

%Evaluate a dataset
% load('linear_data_all.mat')
% load('circular_data.mat')
load('circular_data6.mat')
load('makeFit_params_linear.mat')

N = numel(t);
X = zeros(13,N);


U = [mot1, mot2, mot3, mot4]';
Xreal = [ posX, posY, posZ, quatW, quatX, quatY, quatZ, velX, velY, velZ, angVelX, angVelY, angVelZ ]';

dt = t(2) - t(1);

X(:,1) = Xreal(:, 1);
for ii = 1:N-1
    X(:,ii+1) = rk4(Xreal(:,ii), U(:,ii), params, dt);
end
errors = (X-Xreal).^2;

mses = mean(errors, 2);
transTerms = [1, 2, 3, 8, 9, 10];
rotTerms = [4, 5, 6, 7, 11, 12, 13];
translationalMSE = sum(mses(transTerms));
rotationalMSE = sum(mses(rotTerms));

totalMSE = sum(mses);

fprintf('Translational MSE: %f\n', translationalMSE);
fprintf('Rotational MSE   : %f\n', rotationalMSE);
fprintf('Total MSE        : %f\n', totalMSE);