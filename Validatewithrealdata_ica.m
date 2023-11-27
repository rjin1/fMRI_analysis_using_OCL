ica_map_path = '.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\gicacomps_STBR.mat';
val_path_fmri = '.\COBRE_icaresaug_thresp0FU_de64_GN8_1m2_AN100_int5\1m3_epoch_44_thresh1_34best_test\';
fmri_path_control = '.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_control_frames\';
fmri_path_patient = '.\datasets\COBRE\COBRE_preprocessed_NIAK_noSlTDandGSCremoval_Re6_FWHM\fmri_patient_frames\';
sub_prefix = 'subject00';
frame_prefix = '_frame';
control_data_name_prefix = 'Data_fmri_control';
patient_data_name_prefix = 'Data_fmri_patient';
ind_niftisave = 1;
N_size = 32;
N_results = 9 + 9 + 8 + 2;
N_pad_og1 = 27;
N_pad_og2 = 32;
N_pad_og3 = 26;
N_comp = 8;

load(ica_map_path);

ica_maps = reshape(Data_3D_fmri, [N_comp, N_size^3]);


val_dir = dir(strcat(val_path_fmri,'Test*'));
N_val = length(val_dir);

MSE_record = zeros(N_val, 1);
Corrcoef_record = zeros(N_val, 1);
R_sq_record = zeros(N_val, 1);
REV_record = zeros(N_val, 1);
Subject_name_record = cell(N_val, 1);
Metric_count = 1;

for i = 1 : N_val
    load(strcat(val_path_fmri, val_dir(i).name));
    subject_status = contains(path_frame, 'control');

    name_ind_s = strfind(path_frame, sub_prefix);
    frame_ind_s = strfind(path_frame, frame_prefix);
    subject_name = path_frame(name_ind_s : frame_ind_s - 1);
    frame_name = path_frame(name_ind_s : end - 4);                                    

    if subject_status
        load(strcat(fmri_path_control, control_data_name_prefix, '_mask_',  subject_name, '.mat'));
        load(strcat(fmri_path_control, control_data_name_prefix, '_metainfo_',  subject_name, '.mat'));
    else
        load(strcat(fmri_path_patient, patient_data_name_prefix, '_mask_',  subject_name, '.mat'));
        load(strcat(fmri_path_patient, patient_data_name_prefix, '_metainfo_',  subject_name, '.mat'));
    end

    x_input_resh = reshape(x_input, [N_size^3, 1]);
   [coef_est, ~] = BackReconstructFromDualReg(ica_maps, x_input_resh);
%     coef_est = x_input_resh' * pinv(ica_maps);
    x_recon = coef_est * ica_maps;
    x_recon = reshape(x_recon, [N_size, N_size, N_size]);
    
    V_all = zeros(N_size, N_size, N_size, N_results);
    V_all(:, :, :, 19) = squeeze(Data_3D_fmri(1,:,:,:,:)) * coef_est(1);
    V_all(:, :, :, 20) = squeeze(Data_3D_fmri(2,:,:,:,:)) * coef_est(2);
    V_all(:, :, :, 21) = squeeze(Data_3D_fmri(3,:,:,:,:)) * coef_est(3);
    V_all(:, :, :, 22) = squeeze(Data_3D_fmri(4,:,:,:,:)) * coef_est(4);
    V_all(:, :, :, 23) = squeeze(Data_3D_fmri(5,:,:,:,:)) * coef_est(5);
    V_all(:, :, :, 24) = squeeze(Data_3D_fmri(6,:,:,:,:)) * coef_est(6);
    V_all(:, :, :, 25) = squeeze(Data_3D_fmri(7,:,:,:,:)) * coef_est(7);
    V_all(:, :, :, 26) = squeeze(Data_3D_fmri(8,:,:,:,:)) * coef_est(8);
    V_all(:, :, :, 27) = squeeze(x_input);
    V_all(:, :, :, 28) = squeeze(x_recon);    
    
    V_all_depad = V_all(1:N_pad_og1, 1:N_pad_og2, 1:N_pad_og3, :);
    V_all_final = V_all_depad .* double(Data_mask);
    V_all_size = size(V_all_final);
    
    data_resh_input = reshape(V_all_final(:, :, :, 27), [V_all_size(1)*V_all_size(2)*V_all_size(3), 1]);
    data_resh_recon = reshape(V_all_final(:, :, :, 28), [V_all_size(1)*V_all_size(2)*V_all_size(3), 1]);
    % MSE
    err_fmri = data_resh_input - data_resh_recon;
    mse_fmri = mean(err_fmri.*err_fmri);
    MSE_record(Metric_count) = mse_fmri;
    % Coefficient of determination
    R_sq = 1 - sum(err_fmri.*err_fmri) / sum((data_resh_input - mean(data_resh_input)).^2);
    R_sq_record(Metric_count) = R_sq;
    % Ratio of explained variance
    REV = sum((data_resh_recon - mean(data_resh_input)).^2) / sum((data_resh_input - mean(data_resh_input)).^2);
    REV_record(Metric_count) = REV;
    % Correlation coefficient
    corrcoef_fmri = corrcoef(data_resh_input, data_resh_recon);
    Corrcoef_record(Metric_count) = abs(corrcoef_fmri(1, 2));
    % Record subject frames
    Subject_name_record{Metric_count} = frame_name;
    
    Metric_count = Metric_count + 1;
    
    if ind_niftisave
        Data_fmri_metainfo.ImageSize = [Data_fmri_metainfo.ImageSize(1:3), N_results];
        Data_fmri_metainfo.PixelDimensions = [Data_fmri_metainfo.PixelDimensions(1:3), 1];
        niftiwrite(V_all_final, strcat(val_path_fmri,'ica_', frame_name), Data_fmri_metainfo);
    end
    fprintf(strcat('Finish: MSE and spatial correlation for ', frame_name, '. \n'));
     
end
save(strcat(val_path_fmri, 'icavalresults.mat'), 'MSE_record', 'Corrcoef_record', 'R_sq_record', 'REV_record', 'Subject_name_record');