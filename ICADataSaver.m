% file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Test\';
file_path_gt = '.\datasets\SyntheticFMRI_atl2m8_Ber_192021_new_spread\Volumes\Train\';
N_start = 1;
N_length =69999;
N_sqrtv = 64;

% Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_test#*'));
Files_gt = dir(strcat(file_path_gt, 'FMRISyntheticData_train#*'));
X_input = zeros(N_start+N_length, N_sqrtv * N_sqrtv);
for k = N_start : N_start+N_length
    load(strcat(file_path_gt, Files_gt(k).name));
%     X_temp = reshape(Data_2D_test, [1, N_sqrtv * N_sqrtv]);
    X_temp = reshape(Data_2D_train, [1, N_sqrtv * N_sqrtv]);
    X_input(k,:) = X_temp;
end
save(strcat(file_path_gt, 'X_input.mat'), 'X_input', '-v7.3');