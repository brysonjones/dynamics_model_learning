function addPlots(pos, vel, acc, t, name, ind)

if ~exist('ind', 'var')
    ind = 0; 
    figure;
end

tStep = t(2) - t(1);
posDot     = doDiff(pos, tStep);
posDotDot  = doDiff(posDot, tStep);
velDot     = doDiff(vel, tStep);

% Pos plotting
if ind
    subplot(3,3,ind)
else
    subplot(3,1, 1)
end
plot(t, pos, 'k')
xlabel('Time (s)')
legend('Pos')
ylabel('m', 'FontSize', 14)
title(['Pos ' name], 'Fontsize', 16)

% Vel plotting
if ind
    subplot(3,3,ind + 3)
else
    subplot(3,1, 2)
end
hold all;
plot(t, posDot, 'k--')
plot(t, vel, 'g')
xlabel('Time (s)')
ylabel('m/s', 'FontSize', 14)
legend('d/dt(Pos)', 'Vel')
title(['Vel ' name], 'Fontsize', 16)

% Acc plotting
if ind
    subplot(3,3,ind + 6)
else
    subplot(3,1, 3)
end
hold all;
plot(t, posDotDot, 'k--')
plot(t, velDot, 'g--')
plot(t, acc, 'r')
ylabel('m/s^2', 'FontSize', 14)
legend('d2/dt2(Pos)', 'd/dt(Vel)', 'Acc')
title(['Acc ' name], 'Fontsize', 16)
xlabel('Time (s)', 'Fontsize', 14)