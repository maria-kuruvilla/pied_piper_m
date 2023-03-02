% %% Plot interactions over time
% figure
% %we pick some specific connection strengths to visualize
% pairs=["chinook0 hatchery","chinook0 wild","chinook0 wild to chinook0 hatchery","chinook0 hatchery to chinook0 wild"];
% p=zeros(4,size(J,3));
% p(1,:) = rc_data(1,1:resparams.train_length);
% p(2,:) = rc_data(2,1:resparams.train_length);
% %p(3,:)=J(1,2,:);
% %p(4,:)=J(2,1,:);
% p(3,:)=J(1,2,:)*(max(rc_data(2,:))/max(rc_data(1,:)));%chinook0 wild (2) to chinook0 hatchery (1)
% 
% p(4,:)=J(2,1,:)*(max(rc_data(1,:))/max(rc_data(2,:)));%chinook0 hatchery to chinook0 wild
% 
% 
% for iline=1:4
% subplot(2,2,iline)
% plot(p(iline,:),'-','LineWidth',2)
% xlabel('Time step')
% ylabel('Interaction Strength')
% set(gca,'FontSize',15)
% title(pairs(iline))
% end
% sgtitle('Time-dependent interaction strengths')
% 


%rescaling conn1
conn1_scaled = zeros(1,4,15,15);
for i=1:15
    for j=1:15

        conn1_scaled(:,:,i,j) = compare(:,:,i,j)*(max(x(j,:))/max(x(i,:)));

    end
end

figure
for idel1=1:4
disp(idel1)
subplot(2,2,idel1)
conn1=zeros(15);
%conn1(1:15,1:15)=compare(1,idel1,1:15,1:15);
conn1(1:15,1:15) = conn1_scaled(1,idel1,1:15,1:15);
imagesc(conn1-diag(diag(conn1)))
pbaspect([1 1 1])
xticks(1:1:15)
yticks(1:1:15)
xticklabels(names)
yticklabels(names)
xtickangle(45)
ytickangle(45)
title(dl_labels(1,idel1))
colorbar
set(gca,'FontSize',10)
end