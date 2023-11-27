% addpath('.\LinearICA\Informax');
clear all;clc
file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Test_twotoeight\';
% file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Train\';
file_path_es = '.\InfomaxResults\';
file_path_esMONet = '.\checkpoints_atmosteight_Ber_al2192021_Lunet_bs32_323216_5em1_5em1_int3_85_8435_0.63\experiment_name\epoch85_2t8\';
load('.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\mask.mat');
load(strcat(file_path_es,'S_hat_70000.mat'));
k = 1001; %74,865,68,628,555,107,968,497
N_sqrtv = 64;


Files_es_MONet = dir(strcat(file_path_esMONet, 'Testresults#*.mat'));
[~, reindex] = sort( str2double( regexp( {Files_es_MONet.name}, '\d+', 'match', 'once' )));
Files_es_MONet = Files_es_MONet(reindex);
load(strcat(file_path_esMONet, Files_es_MONet(k).name));
x_com = zeros(8, N_sqrtv, N_sqrtv); % Only test 8 components case.
x_com(1,:,:) = double(squeeze(x0));
x_com(2,:,:) = double(squeeze(x1));
x_com(3,:,:) = double(squeeze(x2));
x_com(4,:,:) = double(squeeze(x3));
x_com(5,:,:) = double(squeeze(x4));
x_com(6,:,:) = double(squeeze(x5));
x_com(7,:,:) = double(squeeze(x6));
x_com(8,:,:) = double(squeeze(x7));
ESMONET_temp = reshape(x_com, [8, N_sqrtv * N_sqrtv]);
ESMONET_temp = ESMONET_temp .* repmat(mask, [8,1]);
m_com(1,:,:) = squeeze(m0);
m_com(2,:,:) = squeeze(m1);
m_com(3,:,:) = squeeze(m2);
m_com(4,:,:) = squeeze(m3);
m_com(5,:,:) = squeeze(m4);
m_com(6,:,:) = squeeze(m5);
m_com(7,:,:) = squeeze(m6);
m_com(8,:,:) = squeeze(m7);
m_com(9,:,:) = squeeze(m8);
x_recon(1,:,:) = squeeze(x_input);
x_recon(2,:,:) = squeeze(x_tilde);
x_recon = reshape(x_recon,[2, N_sqrtv * N_sqrtv]);

Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_test#*'));
% load(strcat(file_path_gt, Files_gt(k).name));
load(strcat(file_path_gt, Files_gt(5).name));
X_temp = reshape(Data_2D_test, [1, N_sqrtv * N_sqrtv]);
X_temp_max = max(abs(X_temp));
X_temp = X_temp / X_temp_max;
GT_temp = zeros(8, N_sqrtv * N_sqrtv);
GT_temp(1:size(Data_2D_test_template, 1),:) = reshape(Data_2D_test_template, [size(Data_2D_test_template, 1), N_sqrtv * N_sqrtv]) / X_temp_max;
S_hat = S_hat .* repmat(mask, [8,1]);
A_temp = X_temp * S_hat' * inv(S_hat * S_hat');
ES_temp = A_temp' .* S_hat;
x_recon(3,:) = sum(ES_temp);

mse_gtesinfomax = zeros(8,8);
mse_gtesMONet = zeros(8,8);
for i_es = 1:8
    for i_gt = 1:8
        mse_gtesinfomax(i_es,i_gt) = norm(GT_temp(i_gt,:) - ES_temp(i_es,:))^2 / N_sqrtv * N_sqrtv;
        mse_gtesMONet(i_es,i_gt) = norm(GT_temp(i_gt,:) - ESMONET_temp(i_es,:))^2 / N_sqrtv * N_sqrtv;
    end
end
LAPpairs_info = matchpairs(mse_gtesinfomax,1e6);
LAPpairs_MONet = matchpairs(mse_gtesMONet,1e6);
LAPMSE_info = zeros(8,1);
LAPMSE_MONet = zeros(8,1);
for i = 1:8
    LAPMSE_info(i) = norm(ES_temp(LAPpairs_info(i,1),:) - GT_temp(LAPpairs_info(i,2),:))^2 / N_sqrtv * N_sqrtv;
    LAPMSE_MONet(i) = norm(ESMONET_temp(LAPpairs_MONet(i,1),:) - GT_temp(LAPpairs_MONet(i,2),:))^2 / N_sqrtv * N_sqrtv;
end
ES_temp = ES_temp(LAPpairs_info(:,1),:);
ESMONET_temp = ESMONET_temp(LAPpairs_MONet(:,1),:);
m_com(1:8,:) = m_com(LAPpairs_MONet(:,1),:);
