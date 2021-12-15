figure
hold all
plot(accZ, 'k')
plot(crossTerms(:, 3))
plot(gTerms(:, 3))
plot(fTerms(:, 3))
plot(acc_pred(:, 3), 'k--')
plot(sum(motors, 2)/200)
plot(accZ - gTerms(:, 3) + crossTerms(:, 3))
legend('Zacc','Cross Term',  'grav', 'F', 'acc pred', 'motor sum', 'otherSide')

%%
figure
subplot(1,3,1)
hold all
plot(t, accX)
plot(t, crossTerms(:, 1))
plot(t, gTerms(:, 1))
plot(t, fTerms(:, 1))
plot(t, acc_pred(:, 1))
legend('Acc', 'Cross', 'Grav', 'Force', 'Acc_pred')

subplot(1,3,2)
hold all
plot(t, accY)
plot(t, crossTerms(:, 2))
plot(t, gTerms(:, 2))
plot(t, fTerms(:, 2))
plot(t, acc_pred(:, 2))
legend('Acc', 'Cross', 'Grav', 'Force', 'Acc_pred')

subplot(1,3,3)
hold all
plot(t, accZ)
plot(t, crossTerms(:, 3))
plot(t, gTerms(:, 3))
plot(t, fTerms(:, 3))
plot(t, acc_pred(:, 3))
legend('Acc', 'Cross', 'Grav', 'Force', 'Acc_pred')