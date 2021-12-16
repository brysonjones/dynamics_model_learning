X = categorical({'Total - Circles', 'Trans. - Circles', 'Rot. - Circles', 'Total - Linear', 'Trans. - Linear', 'Rot. - Linear' });
X = reordercats(X,{'Total - Circles', 'Trans. - Circles', 'Rot. - Circles', 'Total - Linear', 'Trans. - Linear', 'Rot. - Linear' });
Y = [0.000901158, 0.00028728 0.00136157 0.00136157 0.000313626 0.00205458;
    0.000271669827645804  0.000525789666137987 0.0000538528232239326 0.000372053007801964  0.000767716970559075 0.0000329124682958699];
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
title('Next State MSE for White and Black Box')
legend('White Box', 'Black Box', 'Location', 'Southwest')