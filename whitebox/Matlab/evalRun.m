%Evaluate a dataset

N = numel(t);
X = zeros(13,N);


U = [mot1, mot2, mot3, mot4]';
Xreal = [ posX, posY, posZ, quatW, quatX, quatY, quatZ, velX, velY, velZ, angVelX, angVelY, angVelZ ]';

params = [kt_calc; km_calc; J_calc(:)];
dt = t(2) - t(1);

X(:,1) = Xreal(:, 1);
for ii = 1:N-1
    X(:,ii+1) = rk4(X(:,ii), U(:,ii), params, dt);
end

figure
hold all
plot3(Xreal(1,:), Xreal(2, :), Xreal(3, :), 'k')
plot3(X(1,:), X(2, :), X(3, :), 'g')
plot3(Xreal(1,end), Xreal(2, end), Xreal(3, end), 'ko')
plot3(X(1,end), X(2, end), X(3, end), 'go')
legend('Real', 'Pred')
xlabel('X')
ylabel('Y')
zlabel('Z')