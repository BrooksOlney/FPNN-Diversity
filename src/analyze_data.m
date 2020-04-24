files = dir('results/*.csv');
files = {files.name};
n_files = length(files);
percents = zeros(1, n_files);
data = zeros(n_files, 1000, 9);
%ModelA_loss,ModelA_accuracy,PoisonedModelA_loss,PoisonedModelA_accuracy,ModelAB_hamming,ModelB_loss,ModelB_accuracy,PoisonedModelB_loss,PoisonedModelB_accuracy

for i=1:n_files
   percents(i) = str2double(extractBetween(files(i), 'results_', '.csv'))/10;
   data(i, :, :) = csvread(char(files(i)), 1, 0); 
end

% data = cell2mat(data);
% for all files i want:
% average HD
% average accuracy of B
% average poisoned accuracy of B
% difference of accuracy between A -> B
% difference of accuracy between poisoned A -> poisoned B
results = zeros(n_files, 5);

for i=1:n_files
    modelA_acc = data(i, 1, 2) * 100;
    modelApoisoned_acc = data(i, 1, 4) * 100;
    diff_A = abs(modelA_acc - modelApoisoned_acc);
    results(i, 1) = mean(data(i, :, 5)) * 100;
    results(i, 2) = mean(data(i, :, 7)) * 100;
    results(i, 3) = mean(data(i, :, 9)) * 100;
    results(i, 4) = abs(modelA_acc - results(i, 2));
    results(i, 5) = abs(diff_A - abs(results(i, 2) - results(i,3)));
end

HD_plot = figure(1);
ax1 = axes('Parent', HD_plot);
plot(percents, results(:,1));
xlabel(ax1, 'Weight Shift Threshold (%)');
ylabel(ax1, 'Hamming Distance (%)');
title('Average Hamming Distance in Weights');

acc_plot = figure(2);
ax2 = axes('Parent', acc_plot);
plot(percents, results(:,4));
xlabel(ax2, 'Weight Shift Threshold (%)');
ylabel(ax2, 'Difference in Accuracy (%)');
title('Average Difference in Accuracy from A->B');

posioned_plot = figure(3);
ax3 = axes('Parent', posioned_plot);
plot(percents, results(:,5));
xlabel(ax3, 'Weight Shift Threshold (%)');
ylabel(ax3, 'Difference in Accuracy (%)');
title('Average Difference in Accuracy from Poisoning A and B');



% [h, p, ci, stats] = ttest(modelb, modela(1), 'Tail','both')
% [h, p, ci, stats] = ttest(modelb_p, modela_p(1), 'Tail','left')
