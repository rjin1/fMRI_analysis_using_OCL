%X: Input data with the dimension of subjects by voxels.
%D_reduce: order for ICA.
file_path = '.\datasets\COBRE_32\Volumes\Train\';

Files = dir(file_path);
|addpath('.\LinearICA\Informax\');


X = X_input;

D_reduce = 8;
rng(0)
V = icatb_calculate_pca(X',D_reduce, 'whiten', false); 
pca_X = V'*X;

N_run=1;
[N_comp, N_voxel] = size(pca_X);
% W_allruns = zeros(N_comp,N_comp,N_run);
% Shat_allruns = zeros(N_comp,N_voxel,N_run);

for i=1:N_run
    [weights,sphere] = icatb_runica(pca_X); 
    W = weights*sphere;
    S_hat = W*pca_X;
%     A_hat_alldata = inv(W);
%     W_allruns(:,:,i) = W;
%     Shat_allruns(:,:,i) = S_hat;
%     fn=fullfile('Infomax_fMRI_2p0nonlinear_100inits_results',strcat('\linearICAresults_Initialization#',num2str(i),'.mat'));
%     save(fn,'S_hat');
%     fprintf(strcat('Done with run', num2str(i),'\n'));
end
save('.\InfomaxResults\S_hat_70000.mat','S_hat');
% consistentRunidx= RunSelection_crossISIidx(W_allruns);
% fprintf(strcat('Cross-ISI: the most consistent run is #', num2str(consistentRunidx),'\n'));

% Correlation analysis:
% corr = zeros(N_comp, N_comp,N_run);
% for i = 1: N_run
%     corr_temp = corrcoef([Shat_allruns(:,:,i)',SM_mask']);
%     corr(:,:,i) = corr_temp(1:N_comp,N_comp+1:end);
% end
% 
% mean_corr = zeros(N_run,1);
% 
% for i = 1:N_run
%     LAPpairs = matchpairs(1 - corr(:,:,i),1);
%     LAPcorr = zeros(N_comp,1);
%     for j = 1:N_comp
%         LAPcorr(j,1) = corr(LAPpairs(j,1),LAPpairs(j,2),i);
%     end
%     mean_corr(i,1) = mean(LAPcorr);
% end
% max(mean_corr)
%         
%         
% 
% 
