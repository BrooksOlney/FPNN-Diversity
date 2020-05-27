ABfiles = dir('A_to_B/*.csv');
Bfiles = dir('B_poisoned/*.csv');
ABfiles = fullfile({ABfiles.folder}, {ABfiles.name});
Bfiles = fullfile({Bfiles.folder}, {Bfiles.name});

n_files = length(ABfiles);
percents = zeros(1, n_files);
ABdata = zeros(n_files, 1000, 5);
Bdata = zeros(n_files, 999*30, 4);
%ModelA_loss,ModelA_accuracy,PoisonedModelA_loss,PoisonedModelA_accuracy,ModelAB_hamming,ModelB_loss,ModelB_accuracy,PoisonedModelB_loss,PoisonedModelB_accuracy

for i=1:n_files
   percents(i) = str2double(extractBetween(ABfiles(i), 'results_', '.csv'))/10;
   ABdata(i, :, :) = csvread(char(ABfiles(i)), 1, 0); 
   Bdata(i, :, :) = csvread(char(Bfiles(i)), 1, 0);
end

% for all files i want:
% average HD
% average accuracy of B
% average poisoned accuracy of B
% difference of accuracy between A -> B
% difference of accuracy between poisoned A -> poisoned B
ABresults = zeros(n_files, 5);
Bresults = zeros(n_files, 2);

for i=1:n_files
    modelA_acc = ABdata(i, 1, 1) * 100;
    modelApoisoned_acc = ABdata(i, 1, 2) * 100;
    diff_A = abs(modelA_acc - modelApoisoned_acc);
    ABresults(i, 1) = mean(ABdata(i, :, 3)) * 100;
    ABresults(i, 2) = mean(ABdata(i, :, 4)) * 100;
    ABresults(i, 3) = mean(ABdata(i, :, 5)) * 100;
    ABresults(i, 4) = mean(modelA_acc - ABresults(i, 2));
    ABresults(i, 5) = abs(modelApoisoned_acc - ABresults(i,3));
end

for i=1:n_files
    chosenB_acc = mean(Bdata(i, :, 1)) * 100;
    chosenB_pacc = mean(Bdata(i, :, 2)) * 100;
    otherB_acc = mean(Bdata(i, :, 3)) * 100;
    otherB_pacc = mean(Bdata(i, :, 4)) * 100;
    
    Bresults(i, 1) = mean(Bdata(i, :, 1) - Bdata(i, :, 2)) * 100;
    Bresults(i, 2) = mean(Bdata(i, :, 3) - Bdata(i, :, 4)) * 100;
    
%     Bresults(i, 1) = abs(chosenB_acc - chosenB_pacc)
%     Bresults(i, 2) = abs(otherB_acc - otherB_pacc)
end


HD_plot = figure(1);
ax1 = axes('Parent', HD_plot);
plot(percents, ABresults(:,1));
xlabel(ax1, 'Weight Shift Threshold (%)');
ylabel(ax1, 'Hamming Distance (%)');
title('Average Hamming Distance in Weights');

acc_plot = figure(2);
ax2 = axes('Parent', acc_plot);
% plot(percents, results(:,4));
N = size(ABresults, 1);
yMean = mean(ABresults(:,4));
ySEM = std(ABresults(:,4)) / sqrt(N);
CI95 = tinv([0.025 0.975], N-1);
yCI95 = bsxfun(@times, ySEM, CI95(:));

original = plot(percents, ABresults(:,4), 'linewidth', 2.0);
xplot = [percents, fliplr(percents)];
y_plot=[(ABresults(:,4)+yCI95(1))', flipud((ABresults(:,4)+yCI95(2)))'];
hold on
fill(xplot, y_plot, 1, 'facecolor', uint8([17 17 17]), 'edgecolor', 'none', 'facealpha', 0.2);
% plot(percents, results(:,4) + yCI95(1), percents, results(:,4) + yCI95(2));
% hold on
% plot(percents, results(:,5));
hold off
hold on
yMean = mean(ABresults(:,5));
ySEM = std(ABresults(:,5)) / sqrt(N);
CI95 = tinv([0.025 0.975], N-1);
yCI95 = bsxfun(@times, ySEM, CI95(:));
poisoned = plot(percents, ABresults(:,5), 'linewidth', 2.0);
y_plot=[(ABresults(:,5)+yCI95(1))', flipud((ABresults(:,5)+yCI95(2)))'];
fill(xplot, y_plot, 1, 'facecolor', uint8([17 17 17]), 'edgecolor', 'none', 'facealpha', 0.2);

legend([original poisoned], 'Original', 'Poisoned', 'Location', 'NorthWest');
xlabel(ax2, 'Weight Shift Threshold (%)');
ylabel(ax2, 'Difference in Accuracy (%)');
title('Average Difference in Accuracy from A->B');
hold off

% 
% posioned_plot = figure(3);
% ax3 = axes('Parent', posioned_plot);
% plot(percents, results(:,5));
% % hold on
% % plot(percents, results(:,4));
% % hold off
% xlabel(ax3, 'Weight Shift Threshold (%)');
% ylabel(ax3, 'Difference in Accuracy (%)');
% title('Average Difference in Accuracy from Poisoning A and B');

B_plot = figure(3);
ax3 = axes('Parent', B_plot);
b_orig = plot(percents, Bresults(:,1));
hold on
b_poisoned = plot(percents, Bresults(:, 2));
legend([b_orig b_poisoned], 'Chosen B', 'Other Bs', 'Location', 'NorthWest');
xlabel(ax3, 'Weight Shift Threshold (%)');
ylabel(ax3, 'Weight Difference(%)');
title('Chosen B vs Non-Chosen Bs');

% [h, p, ci, stats] = ttest(modelb, modela(1), 'Tail','both')
% [h, p, ci, stats] = ttest(modelb_p, modela_p(1), 'Tail','left')
