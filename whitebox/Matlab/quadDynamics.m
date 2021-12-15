function xdot = quadDynamics(x, motors, params)
%
% input
%   x[1:3]   - posN
%   x[4:7]   - quat
%   x[8:10]  - velB
%   x[11:13] - omega

H = zeros(4,3);
H(2:end, :) = eye(3);
posN = x(1:3);
quat = x(4:7);
velB = x(8:10);
omega = x(11:13);

rotm = quat2rotm(quat')'; %needs row vec, our equations use different def of rotMat

kt = params(1);
km = params(2);
J = reshape(params(3:end), 3, 3);

m = 0.752;
sLen = 0.126;
g = 9.81;
gVec = [0; 0; -g];


FMat   = [0, 0, 0, 0;
          0, 0, 0, 0;
          kt, kt, kt, kt];
tauMat = [-sLen*kt,       0,        0,  sLen*kt;
          0,        -sLen*kt,  sLen*kt,       0;
          -km,           km,       km,      -km];

fTerm = FMat * motors(:)/m;
gTerm = rotm' * gVec;
crossTerm = cross(omega, velB);
acc_pred = fTerm + gTerm - crossTerm; %-g + cross?

%Need to do the angular in the principal then rotate
Jw = J * omega;

angAcc_pred = J \ (tauMat*motors - cross(omega, Jw));

xdot = [rotm * velB;
    1/2 * L(quat) * H * omega;
    acc_pred;
    angAcc_pred];
end

function Lmat = L(quat)
qHat = hat(quat);

L = zeros(4,4);
L(1, 1)     = quat(1);
L(1, 2:end) = -1 * quat(2:end);
L(2:end, 1) = quat(2:end);
L(2:end, 2:end) = quat(1) * eye(3) + qHat;

Lmat = L;
end

function qHat = hat(quat)
qHat = [0, -quat(4), quat(3);
    quat(4), 0, -quat(2);
    -quat(3), quat(2), 0];
end