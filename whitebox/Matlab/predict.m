function [acc_pred, angAcc_pred, fTerms, gTerms, crossTerms] = predict(motors, rotms, omegas, velB, omegasPrinc, kt, km, J)

g = 9.81;
m = 0.752;
sLen = .126;

angleVal = pi/4;
bod2princRotMat = [cos(angleVal), -sin(angleVal), 0;
         sin(angleVal), cos(angleVal), 0;
          0, 0, 1];

numPts = size(motors, 1);

gVec = [0; 0; -g*m];
acc_pred = zeros(numPts, 3);
angAcc_pred = zeros(numPts, 3);

FMat   = [0, 0, 0, 0;
          0, 0, 0, 0;
          kt, kt, kt, kt];
tauMat = [-sLen*kt,       0,        0,  sLen*kt;
          0,        -sLen*kt,  sLen*kt,       0;
          -km,           km,       km,      -km];
fTerms = zeros(numPts, 3);
gTerms = zeros(numPts, 3);
crossTerms = zeros(numPts, 3);

for ii = 1:numPts
    fTerms(ii, :) = FMat * motors(ii, :)'/m;
    gTerms(ii, :) = rotms{ii}' * gVec/m;
    crossTerms(ii, :) = cross(omegas(ii,:), velB(ii,:));
    acc_pred(ii,:) = fTerms(ii, :)' + gTerms(ii, :)' - crossTerms(ii, :)'; %-g + cross?
    
    %Need to do the angular in the principal then rotate
    thisOmegaPrinc = omegasPrinc(ii,:)';
    Jw = J * thisOmegaPrinc;
    angAccPrinc_pred = J \ (tauMat*motors(ii,:)' - cross(thisOmegaPrinc, Jw));
    angAcc_pred(ii,:) = bod2princRotMat'*angAccPrinc_pred;
end