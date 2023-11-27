for i = 499
path_fmri = '.\COBRE32FU_bs32_32_64_16_5em1_S0_5em1_S0_N500_S0corr_int5\epoch165_val\';
file_name = strcat('Testresults#',num2str(i),'.mat'); %425, 250
N_results = 9 + 9 + 9 + 9 + 8 + 2;
N_size = 32;

load(strcat(path_fmri, file_name));
subject_frame_name = path_frame(strfind(path_frame, 'subject') : end);
subject_name = subject_frame_name(1 : strfind(subject_frame_name, '_') - 1);
path_metainfo = strcat('.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_control_frames\Data_fmri_control_metainfo_', subject_name, '.mat');
% path_metainfo = '.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_patient_frames\Data_fmri_patient_metainfo_' + subject_name + '.mat';
path_mask = strcat('.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_control_frames\Data_fmri_control_mask_', subject_name, '.mat');
% path_mask = '.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_patient_frames\Data_fmri_control_mask_' + subject_name + '.mat';
save_name = strcat('Results_', subject_frame_name);
padding = 1;
N_pad_og1 = 27;
N_pad_og2 = 32;
N_pad_og3 = 26;

load(path_metainfo);
load(path_mask);

Data_fmri_metainfo.ImageSize = [Data_fmri_metainfo.ImageSize(1:3), N_results];
Data_fmri_metainfo.PixelDimensions = [Data_fmri_metainfo.PixelDimensions(1:3), 1];

V_all = zeros(N_size, N_size, N_size, N_results);

V_all(:, :, :, 1) = squeeze(m0);
V_all(:, :, :, 2) = squeeze(m1);
V_all(:, :, :, 3) = squeeze(m2);
V_all(:, :, :, 4) = squeeze(m3);
V_all(:, :, :, 5) = squeeze(m4);
V_all(:, :, :, 6) = squeeze(m5);
V_all(:, :, :, 7) = squeeze(m6);
V_all(:, :, :, 8) = squeeze(m7);
V_all(:, :, :, 9) = squeeze(m8);
V_all(:, :, :, 10) = squeeze(m_tilde0);
V_all(:, :, :, 11) = squeeze(m_tilde1);
V_all(:, :, :, 12) = squeeze(m_tilde2);
V_all(:, :, :, 13) = squeeze(m_tilde3);
V_all(:, :, :, 14) = squeeze(m_tilde4);
V_all(:, :, :, 15) = squeeze(m_tilde5);
V_all(:, :, :, 16) = squeeze(m_tilde6);
V_all(:, :, :, 17) = squeeze(m_tilde7);
V_all(:, :, :, 18) = squeeze(m_tilde8);
V_all(:, :, :, 19) = squeeze(x0);
V_all(:, :, :, 20) = squeeze(x1);
V_all(:, :, :, 21) = squeeze(x2);
V_all(:, :, :, 22) = squeeze(x3);
V_all(:, :, :, 23) = squeeze(x4);
V_all(:, :, :, 24) = squeeze(x5);
V_all(:, :, :, 25) = squeeze(x6);
V_all(:, :, :, 26) = squeeze(x7);
V_all(:, :, :, 27) = squeeze(x_input);
V_all(:, :, :, 28) = squeeze(x_tilde);
% V_all(:, :, :, 29) = squeeze(s0);
% V_all(:, :, :, 30) = squeeze(s1);
% V_all(:, :, :, 31) = squeeze(s2);
% V_all(:, :, :, 32) = squeeze(s3);
% V_all(:, :, :, 33) = squeeze(s4);
% V_all(:, :, :, 34) = squeeze(s5);
% V_all(:, :, :, 35) = squeeze(s6);
% V_all(:, :, :, 36) = squeeze(s7);
% V_all(:, :, :, 37) = squeeze(s8);
% V_all(:, :, :, 38) = squeeze(alpha0);
% V_all(:, :, :, 39) = squeeze(alpha1);
% V_all(:, :, :, 40) = squeeze(alpha2);
% V_all(:, :, :, 41) = squeeze(alpha3);
% V_all(:, :, :, 42) = squeeze(alpha4);
% V_all(:, :, :, 43) = squeeze(alpha5);
% V_all(:, :, :, 44) = squeeze(alpha6);
% V_all(:, :, :, 45) = squeeze(alpha7);
% V_all(:, :, :, 46) = squeeze(alpha8);


if  padding ==1
    V_all_depad = V_all(1:N_pad_og1, 1:N_pad_og2, 1:N_pad_og3, :);
    V_all_final = V_all_depad .* double(Data_mask);
else
    V_all_final = V_all .* double(Data_mask);
end

niftiwrite(V_all_final, strcat(path_fmri, save_name), Data_fmri_metainfo);
clear all;clc
end