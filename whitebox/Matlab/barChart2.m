X = categorical({'Total MSE', 'Trans. MSE', 'Rot.  MSE'});
X = reordercats(X,{'Total MSE', 'Trans. MSE', 'Rot.  MSE'});
Y = [0.00201807617  0.004708745502 7.43E-08;
    0.1430387081  7.77E-09 0.2503177351]
bar(X,Y)

%Add boxes with values
xAx = 0.25*(-1:1)' + (1:6);
for k=1:size(Y,1)
    for m = 1:size(Y,2)
        h = text(xAx(k,m),Y(k,m),num2str(Y(k,m),'%0.2g'),...
            'HorizontalAlignment','left',...
            'VerticalAlignment','bottom', 'FontSize', 8);
%         set(h,'Rotation',-45);
    end
end


set(gca, 'YScale', 'log')
ylabel('Next State MSE')
ylim([1e-9, 1])
title('Next State MSE for Gray Box')
legend('Linear', 'Circular', 'Location', 'Southwest')