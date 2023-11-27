addpath ./simtb_v18/sim
addpath ./plot_code

% Parameters for data generation
nV = 64;
SM_source_ID = [6, 7, 16, 17, 19, 20, 27, 28];
SM_translate_x = 0;
SM_translate_y = 0;
SM_theta = 0;
SM_spread_unif_start = 0.25;
SM_spread_unif_end = 0.25;% U(0.25,1.75)

seed = 0;
rng(seed)

% Mask generation
arg1 = linspace(-1,1,nV);
[x,y] = meshgrid(arg1,arg1);
r = sqrt(x.^2 + y.^2);
mask = ones(nV,nV);
mask(r>1) = 0;
mask = reshape(mask,1,nV*nV);
mask_2D = reshape(mask, [1, nV, nV]);

[Data, Data_temp]  = data_gen(nV, SM_source_ID, SM_translate_x, SM_translate_y, SM_theta, SM_spread_unif_start, SM_spread_unif_end, mask);
Data_2D_test = reshape(Data, [1, nV, nV]);
Data_2D_test_template = reshape(Data_temp, [length(SM_source_ID), nV, nV]);
% save(strcat(Data_test_path,'FMRISyntheticData_test#',num2str(i),'.mat'),'Data_2D_test','Data_2D_test_template');

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

