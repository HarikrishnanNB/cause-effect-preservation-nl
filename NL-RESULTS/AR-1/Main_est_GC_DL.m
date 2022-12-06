% Passing AR processes with varying coupling/ corresponding firing times of
% ChaosFEX to MVGC for granger causality estimation


close all; clear all;

N_trials=50;
nobs=2000;
nvars=2;
momax=20; %30;

coup=0:0.1:1;

for corr_level=1:11
    
    %nameDataDir  = ['C:\\0 RES\\Code\\2022\\Hari_ICML\\AR-1\\%1.1f\\' num2str(coup(corr_level)) '\'];
    nameDataDir = sprintf('C:\\0 RES\\Code\\2022\\Hari_ICML\\AR-1\\%1.1f\\',coup(corr_level));
    cd(nameDataDir)
    load('class_0_indep_dl_features.mat');
    load('class_1_dep_dl_features.mat');
    Y_ind= class_0_indep_dl_features(1:N_trials,:);
    X_dep= class_1_dep_dl_features(1:N_trials,:);
    clear class_0_indep_dl_features  class_1_dep_dl_features
    
%     load('Y_independent_data_class_0');
%     load('X_dependent_data_class_1');
%     Y_ind= class_0_indep_raw_data(1:N_trials,:);
%     X_dep= class_1_dep_raw_data(1:N_trials,:);
%     clear class_0_indep_raw_data class_1_dep_raw_data
    
    Y_ind2=permute(Y_ind, [3 2 1]);
    X_dep2=permute(X_dep, [3 2 1]);
    
    X1=[Y_ind2; X_dep2];
    [F,sig,pval]=mvgc_main(X1,nvars,nobs,N_trials,momax);
    F_mean_right(corr_level)=F(2,1);
    F_mean_opp(corr_level)=F(1,2);
    sig_right(corr_level)=sig(2,1);
    sig_opp(corr_level)=sig(1,2);
    clear F sig pval X1 Y_ind X_dep Y_ind2 X_dep2
    
end
close all;
plot([0:0.1:1.0],F_mean_right);
hold on; plot([0:0.1:1.0],F_mean_opp, 'r');
legend('Right direction', 'Opp direction');
xlabel('Coupling coefficient');
ylabel('GC (F-statistic)');