X = categorical({'Total - Circles', 'Trans. - Circles', 'Rot. - Circles', 'Total - Linear', 'Trans. - Linear', 'Rot. - Linear' });
X = reordercats(X,{'Total - Circles', 'Trans. - Circles', 'Rot. - Circles', 'Total - Linear', 'Trans. - Linear', 'Rot. - Linear' });
Y = [0.016264, 0.001882, 0.014382, 0.011255, 0.001724, 0.009531;  %0.15906, 0.001814, 0.014092, 
     0.002419, 0.002416, 0.000111282, 0.005086, 0.005003, 0.0000830;  %0.002609, 0.002518, 0.0000908, 
     0.002018, 0.0047087, 7.43e-8, 0.002018, 0.0047087, 7.43e-8];
bar(X,Y)
set(gca, 'YScale', 'log')
ylabel('Next State MSE')
title('Next State MSE for Three Approaches to SI')
legend('White Box', 'Black Box', 'Gray Box', 'Location', 'Southwest')