val_path_fmri = '.\COBRE_icaresaug_thresp0FU_de64_GN8_1m2_AN100_int5_togo\1m3_epoch_44_conti5m3_44_conti1m2_44_conti_1m3_42_8235_togo\';
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
%     V_all(:, :, :, 10) = squeeze(m_tilde0);
%     V_all(:, :, :, 11) = squeeze(m_tilde1);
%     V_all(:, :, :, 12) = squeeze(m_tilde2);
%     V_all(:, :, :, 13) = squeeze(m_tilde3);
%     V_all(:, :, :, 14) = squeeze(m_tilde4);
%     V_all(:, :, :, 15) = squeeze(m_tilde5);
%     V_all(:, :, :, 16) = squeeze(m_tilde6);
%     V_all(:, :, :, 17) = squeeze(m_tilde7);
%     V_all(:, :, :, 18) = squeeze(m_tilde8);
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
        niftiwrite(V_all_final, strcat(val_path_fmri, frame_name), Data_fmri_metainfo);
    end
    fprintf(strcat('Finish: MSE and spatial correlation for ', frame_name, '. \n'));
     
end
save(strcat(val_path_fmri, 'valresults.mat'), 'MSE_record', 'Corrcoef_record', 'R_sq_record', 'REV_record', 'Subject_name_record');