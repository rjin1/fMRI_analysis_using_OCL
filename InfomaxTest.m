% addpath('.\LinearICA\Informax');

file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Test_twotoeight\';
% file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Train\';
file_path_es = '.\InfomaxResults\';
load('.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\mask.mat');
load(strcat(file_path_es,'S_hat_70000.mat'));
N_start = 1;
N_length =999;
N_sqrtv = 64;

Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_test#*'));
% MeanCorrPerSample = zeros(N_start+N_length,1);
MeanMSEPerSample = zeros(N_start+N_length,1);
% Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_train#*'));
for k = N_start : N_start+N_length
    load(strcat(file_path_gt, Files_gt(k).name));
    X_temp = reshape(Data_2D_test, [1, N_sqrtv * N_sqrtv]);
    X_temp_max = max(abs(X_temp));
    X_temp = X_temp / X_temp_max;
    GT_temp = zeros(8, N_sqrtv * N_sqrtv);
    GT_temp(1:size(Data_2D_test_template, 1),:) = reshape(Data_2D_test_template, [size(Data_2D_test_template, 1), N_sqrtv * N_sqrtv]) / X_temp_max;
    S_hat = S_hat .* repmat(mask, [8,1]);
    A_temp = X_temp * S_hat' * inv(S_hat * S_hat');
    ES_temp = A_temp' .* S_hat;
    
    mse_gtes = zeros(8,8);
    for i_gt = 1:8
        for i_es = 1:8
            mse_gtes(i_gt,i_es) = norm(GT_temp(i_gt,:) - ES_temp(i_es,:))^2 / N_sqrtv * N_sqrtv;
        end
    end
%     corr = abs(corrcoef([GT_temp', ES_temp']));
%     corr = corr(1:8,9:16);
%     LAPpairs = matchpairs(1-corr,1);
    LAPpairs = matchpairs(mse_gtes,1e6);
%     LAPcorr = zeros(8,1);
    LAPMSE = zeros(8,1);
    for i = 1:8
%         LAPcorr(i) = corr(LAPpairs(i,1),LAPpairs(i,2));
        LAPMSE(i) = norm(GT_temp(LAPpairs(i,1),:) - ES_temp(LAPpairs(i,2),:))^2 / N_sqrtv * N_sqrtv;
    end
%     MeanCorrPerSample(k) = mean(LAPcorr);
    MeanMSEPerSample(k,1) = mean(LAPMSE);
    MeanMSEPerSample(k,2) = size(Data_2D_test_template, 1);
    save(strcat(file_path_es, 'TestresultsMSE#', num2str(k-1),'.mat'), 'LAPpairs', 'LAPMSE');
    display(k);
end
% mean(MeanCorrPerSample)
mean(MeanMSEPerSample(:,1))

% Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_train#*'));
% Files_es = dir(strcat(file_path_es, 'Testresults#*'));
% [~, reindex] = sort( str2double( regexp( {Files_es.name}, '\d+', 'match', 'once' )));
% Files_es = Files_es(reindex);

% MeanCorrPerSample = zeros(N_start+N_length,1);
% for k = N_start : N_start+N_length
%     load(strcat(file_path_gt, Files_gt(k).name));
%     load(strcat(file_path_es, Files_es(k).name));
%     
%     GT_temp = reshape(Data_2D_test_template, [8, N_sqrtv * N_sqrtv]);
%     x_com = zeros(8, N_sqrtv, N_sqrtv); % Only test 8 components case.
%     x_com(1,:,:) = double(squeeze(x0));
%     x_com(2,:,:) = double(squeeze(x1));
%     x_com(3,:,:) = double(squeeze(x2));
%     x_com(4,:,:) = double(squeeze(x3));
%     x_com(5,:,:) = double(squeeze(x4));
%     x_com(6,:,:) = double(squeeze(x5));
%     x_com(7,:,:) = double(squeeze(x6));
%     x_com(8,:,:) = double(squeeze(x7));
%     ES_temp = reshape(x_com, [8, N_sqrtv * N_sqrtv]);
%     
%     corr = abs(corrcoef([GT_temp', ES_temp']));
%     corr = corr(1:8,9:16);
%     LAPpairs = matchpairs(1-corr,1);
%     LAPcorr = zeros(8,1);
%     for i = 1:8
%         LAPcorr(i) = corr(LAPpairs(i,1),LAPpairs(i,2));
%     end
%     MeanCorrPerSample(k) = mean(LAPcorr);
%     save(strcat(file_path_es, 'TestresultsCorr#', num2str(k-1),'.mat'), 'LAPpairs', 'LAPcorr');
% end
% mean(MeanCorrPerSample)