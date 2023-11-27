addpath ./simtb_v18/sim
addpath ./plot_code

% Parameters for data generation
nV = 64;
SM_source_ID = [6, 7, 16, 17, 19, 20, 27, 28];
SM_translate_x = 0;
SM_translate_y = 0;
SM_theta = 0;
SM_spread_unif_start = 1.75;
SM_spread_unif_end = 1.75;% U(0.25,1.75)

N_sample_train = 70000;
N_sample_valid = 1000;
N_sample_test = 1000;

Data_path = './datasets/SyntheticFMRI_atl2m8_Ber_192021_new_spread/Volumes/';
data_seed = 1;
rng(data_seed)

% Components are activated with equal likelihood. 
Ber_p = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] .* ones(1, length(SM_source_ID));
N_com_least = 2; % At least N_com_least components in each volume.
Index_temp = 0;  

% Mask generation
arg1 = linspace(-1,1,nV);
[x,y] = meshgrid(arg1,arg1);
r = sqrt(x.^2 + y.^2);
mask = ones(nV,nV);
mask(r>1) = 0;
mask = reshape(mask,1,nV*nV);
mask_2D = reshape(mask, [1, nV, nV]);

% Data folder check
if ~exist(Data_path, 'dir')
    mkdir(Data_path)
end

% Save the mask
save(strcat(Data_path,'mask.mat'), 'mask', 'mask_2D');

% Training, validaiton, test generation and save
fprintf('Data generation start\n')
% Data_train_path = strcat(Data_path,'Train/');
% if ~exist(Data_train_path, 'dir')
%     mkdir(Data_train_path)
% end
% for i = 1:N_sample_train
% %     N_sources = randi(length(SM_source_ID));
% %     Index_topick = randperm(length(SM_source_ID));
% %     Index_topick = Index_topick(1:N_sources); 
%     while sum(Index_temp) <= N_com_least - 1
%         Index_temp = binornd(1, Ber_p);
%     end
%     Index_topick = logical(Index_temp);
%     Index_temp = 0;
%     [Data, ~]  = data_gen(nV, SM_source_ID(Index_topick), SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
%     Data_2D_train = reshape(Data, [1, nV, nV]);
% %     Data_2D_train_template = reshape(Data_temp, [length(SM_source_ID), nV, nV]);
%     save(strcat(Data_train_path,'FMRISyntheticData_train#',num2str(i),'.mat'),'Data_2D_train');
%     clear Data Data_temp Data_2D_train Data_2D_train_template
%     if ~rem(i,1000)
%         fprintf(strcat('Training sample#',num2str(i),'\n'));
%     end
% end
% 
% Data_valid_path = strcat(Data_path,'Valid/');
% if ~exist(Data_valid_path, 'dir')
%     mkdir(Data_valid_path)
% end
% for i = 1:N_sample_valid
%     [Data, Data_temp]  = data_gen(nV, SM_source_ID, SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
%     Data_2D_valid = reshape(Data, [1, nV, nV]);
%     Data_2D_valid_template = reshape(Data_temp, [length(SM_source_ID), nV, nV]);
%     save(strcat(Data_valid_path,'FMRISyntheticData_valid#',num2str(i),'.mat'),'Data_2D_valid','Data_2D_valid_template');
%     clear Data Data_temp Data_2D_valid Data_2D_valid_template
%     if ~rem(i,100)
%         fprintf(strcat('Validation sample#',num2str(i),'\n'));
%     end
% end
    
% Data_test_path = strcat(Data_path,'Test_twotoeight/');
% if ~exist(Data_test_path, 'dir')
%     mkdir(Data_test_path)
% end
% for i = 1:N_sample_test
%     [Data, Data_temp]  = data_gen(nV, SM_source_ID, SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
%     Data_2D_test = reshape(Data, [1, nV, nV]);
%     Data_2D_test_template = reshape(Data_temp, [length(SM_source_ID), nV, nV]);
%     save(strcat(Data_test_path,'FMRISyntheticData_test#',num2str(i),'.mat'),'Data_2D_test','Data_2D_test_template');
%     clear Data Data_temp Data_2D_test Data_2D_test_template
%     if ~rem(i,100)
%         fprintf(strcat('Test sample#',num2str(i),'\n'));
%     end
% end
% 

Data_test_path = strcat(Data_path,'Test_twotoeight/');
if ~exist(Data_test_path, 'dir')
    mkdir(Data_test_path)
end
%     [Data, ~]  = data_gen(nV, SM_source_ID(Index_topick), SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
%     Data_2D_train = reshape(Data, [1, nV, nV]);
% %     Data_2D_train_template = reshape(Data_temp, [length(SM_source_ID), nV, nV]);
%     save(strcat(Data_train_path,'FMRISyntheticData_train#',num2str(i),'.mat'),'Data_2D_train');
%     clear Data Data_temp Data_2D_train Data_2D_train_template
%     if ~rem(i,1000)
%         fprintf(strcat('Training sample#',num2str(i),'\n'));
%     end
% end

for i = 1:N_sample_test
    while sum(Index_temp) <= N_com_least - 1
        Index_temp = binornd(1, Ber_p);
    end
    Index_topick = logical(Index_temp);
    Index_temp = 0;
    [Data, Data_temp]  = data_gen(nV, SM_source_ID(Index_topick), SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
    Data_2D_test = reshape(Data, [1, nV, nV]);
    Data_2D_test_template = reshape(Data_temp, [length(SM_source_ID(Index_topick)), nV, nV]);
    save(strcat(Data_test_path,'FMRISyntheticData_test#',num2str(i),'.mat'),'Data_2D_test','Data_2D_test_template');
    clear Data Data_temp Data_2D_test Data_2D_test_template
    if ~rem(i,100)
        fprintf(strcat('Test sample#',num2str(i),'\n'));
    end
end

fprintf('Data generation done\n')

% Synthetic Data generation 
function [Data, Data_temp]  = data_gen(nV, SM_source_ID, SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask) 
    % Source generation
    N_SM = length(SM_source_ID);
    SM = zeros(N_SM, nV*nV);
    for i = 1:N_SM
        SM_spread = rand(1) * (SM_spread_unif_end - SM_spread_unif_start) + SM_spread_unif_start; 
        Temp = simtb_generateSM(SM_source_ID(i), nV, SM_translate_x, SM_translate_y, SM_theta, SM_spread);
        SM(i,:) = mask.*(reshape(Temp,1,nV*nV) + 0.005*randn(1, nV*nV));
        clear Temp 
    end

    % Time course generation
%     tc = rand(1, N_SM) * 2 -1; %U(-1,1) 
%     tc = ones(1, N_SM);
    tc_ber_p = 0.5 * ones(1,N_SM);
    tc = binornd(1, tc_ber_p) * 2 -1;
%     tc = tc.* (rand(1, N_SM) * 0.5 + 0.5);
    Data_temp = tc' .* SM;
    Data = tc * SM;
end




