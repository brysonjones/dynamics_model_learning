% Playing around with data n such
% uiopen('C:\Users\zvick\OneDrive\Documents\GitHub\dynamics_model_learning\processed_data\merged_2021-02-05-14-00-56_seg_1.csv',1)
% 
% circular - '2021-02-05-14-00-56'
% vertical - '2021-02-05-14-19-34'
% linear   - "2021-02-03-16-10-37"

g = 9.81; 
m = 0.752; %kg
sLen = 0.126; %m (from https://armattanquads.com/chameleon-ti-6-inch/)

% useInds = 1:800;
% t       = t(useInds);
% quatW   = quatW(useInds);
% quatX   = quatX(useInds);
% quatY   = quatY(useInds);
% quatZ   = quatZ(useInds);
% posX    = posX(useInds);
% posY    = posY(useInds);
% posZ    = posZ(useInds);
% velX    = velX(useInds);
% velY    = velY(useInds);
% velZ    = velZ(useInds);
% accX    = accX(useInds);
% accY    = accY(useInds);
% accZ    = accZ(useInds);
% angVelX = angVelX(useInds);
% angVelY = angVelY(useInds);
% angVelZ = angVelZ(useInds);
% angAccX = angAccX(useInds);
% angAccY = angAccY(useInds);
% angAccZ = angAccZ(useInds);
% mot1    = mot1(useInds);
% mot2    = mot2(useInds);
% mot3    = mot3(useInds);
% mot4    = mot4(useInds);

numPts = numel(t);
tStep = t(2) - t(1);
quats = [quatW, quatX, quatY, quatZ];

posN = [posX, posY, posZ];
velB = [velX, velY, velZ];
accB = [accX, accY, accZ];

velN = zeros(numPts, 3);
accN = zeros(numPts, 3);
rotms = cell(numPts, 1);

for ii = 1:numPts
    rotm = quat2rotm(quats(ii, :));
    rotms{ii} = rotm'; %becuse their definition is going the other way
    velN(ii, :) = rotm * velB(ii, :)';
    accN(ii, :) = rotm * accB(ii, :)';
end

figure
addPlots(posN(:, 1), velN(:, 1), accN(:, 1),        t, 'X', 1)
addPlots(posN(:, 2), velN(:, 2), accN(:, 2),        t, 'Y', 2)
addPlots(posN(:, 3), velN(:, 3), accN(:, 3) - 9.81, t, 'Z', 3)

figure; hold all
euls = zeros(numPts, 3);
for ii = 1:numPts
    euls(ii, :) = quat2eul(quats(ii, :));
end
plot(t, euls(:, 1))
plot(t, euls(:, 2))
plot(t, euls(:, 3))
xlabel('time (sec)')
ylabel('Angles')
legend('Z', 'Y', 'X')

%% Replicating kt calculations

motors = [mot1, mot2, mot3, mot4];
omegas = [angVelX, angVelY, angVelZ];

aTerms = sum(motors, 2)/m;
bTerms = zeros(numPts, 1);

for ii = 1:numPts
    rotm = rotms{ii};
    tmp = rotm' * [0; 0; -g] - cross(omegas(ii, :), velB(ii,:))';
    bTerms(ii) = tmp(3);
end
otherSide = accB(:, 3) - bTerms;

% figure
% scatter(aTerms, otherSide);

% [coeffs, ~] = polyfit(aTerms, otherSide, 1);
kt = lsqr(aTerms, otherSide)

%% replicate J calculations
angAcc = [angAccX, angAccY, angAccZ];

%rotate into our principal frame
angleVal = pi/4;
bod2princRotMat = [cos(angleVal), -sin(angleVal), 0;
         sin(angleVal), cos(angleVal), 0;
          0, 0, 1];
omegasPrinc = (bod2princRotMat * omegas')';
angAccPrinc = (bod2princRotMat * angAcc')';

%Look at rotation around x first
w1A = angAccPrinc(:,1) + omegas(:,2) .* omegas(:, 3);
w1B = sLen*kt*(motors(:, 4) - motors(:, 1));

%then around y
w2A = angAccPrinc(:,2) - omegas(:,1) .* omegas(:, 3);
w2B = sLen*kt*(motors(:, 3) - motors(:, 2));

% scatter([w1A; w2A], [w1B; w2B])
Jx = lsqr([w1A; w2A], [w1B; w2B]); %Jx = Jy
J = diag([Jx, Jx, 2*Jx])

%% Lastly, the km calculation
a = 1/(2*Jx)*(-mot1 + mot2 + mot3 - mot4);
b = angAcc(:,3);
km = lsqr(a, b)

%% Now we can repredict the accelerations

% km = km_calc;
% kt = kt_calc;
% J = diag(diag(J_calc));
[acc_pred, angAcc_pred] = predict(motors, rotms, omegas, velB, omegasPrinc, kt, km, J);



%% And plot the results
for hiding = 1 %Just to allow me to hide plotting code 
figure
subplot(2,3,1)
hold all
plot(t, accB(:, 1), 'k')
plot(t, acc_pred(:, 1), 'g')
xlabel('t')
ylabel('acc')
title('acc x')

subplot(2,3,2)
hold all
plot(t, accB(:, 2), 'k')
plot(t, acc_pred(:, 2), 'g')
xlabel('t')
ylabel('acc')
title('acc y')

subplot(2,3,3)
hold all
plot(t, accB(:, 3), 'k')
plot(t, acc_pred(:, 3), 'g')
xlabel('t')
ylabel('acc')
title('acc z')

%
subplot(2,3,4)
hold all
plot(t, angAcc(:, 1), 'k')
plot(t, angAcc_pred(:, 1), 'g')
xlabel('t')
ylabel('angAcc')
title('angAcc x')

subplot(2,3,5)
hold all
plot(t, angAcc(:, 2), 'k')
plot(t, angAcc_pred(:, 2), 'g')
xlabel('t')
ylabel('angAcc')
title('angAcc y')

subplot(2,3,6)
hold all
plot(t, angAcc(:, 3), 'k')
plot(t, angAcc_pred(:, 3), 'g')
xlabel('t')
ylabel('angAcc')
title('angAcc z')
end