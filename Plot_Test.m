% close all
% clear all;clc
load Testresults#2.mat %
% load SM.mat SM
% thresh = 1e-3;
m_com(1,:,:) = squeeze(m0);
m_com(2,:,:) = squeeze(m1);
m_com(3,:,:) = squeeze(m2);
m_com(4,:,:) = squeeze(m3);
m_com(5,:,:) = squeeze(m4);
m_com(6,:,:) = squeeze(m5);
m_com(7,:,:) = squeeze(m6);
m_com(8,:,:) = squeeze(m7);
m_com(9,:,:) = squeeze(m8);
m_tilde_com(1,:,:) = squeeze(m_tilde0);
m_tilde_com(2,:,:) = squeeze(m_tilde1);
m_tilde_com(3,:,:) = squeeze(m_tilde2);
m_tilde_com(4,:,:) = squeeze(m_tilde3);
m_tilde_com(5,:,:) = squeeze(m_tilde4);
m_tilde_com(6,:,:) = squeeze(m_tilde5);
m_tilde_com(7,:,:) = squeeze(m_tilde6);
m_tilde_com(8,:,:) = squeeze(m_tilde7);
m_tilde_com(9,:,:) = squeeze(m_tilde8);
% m_com_thresh(1,:,:) = squeeze(m0)>thresh;
% m_com_thresh(2,:,:) = squeeze(m1)>thresh;
% m_com_thresh(3,:,:) = squeeze(m2)>thresh;
x_recon(1,:,:) = squeeze(x_input);
x_recon(2,:,:) = squeeze(x_tilde);
x_com(1,:,:) = squeeze(x0);
x_com(2,:,:) = squeeze(x1);
x_com(3,:,:) = squeeze(x2);
x_com(4,:,:) = squeeze(x3);
x_com(5,:,:) = squeeze(x4);
x_com(6,:,:) = squeeze(x5);
x_com(7,:,:) = squeeze(x6);
x_com(8,:,:) = squeeze(x7);
showSMsyn_modified(m_tilde_com,[1,9],0)
showSMsyn_modified(m_com,[1,9],0)
showSMsyn_modified(x_com,[1,8],1,[-1,1]) %Scaled templates.
showSMsyn_modified(x_com,[1,8],0)
showSMsyn_modified(x_recon,[1,2],0)

% load FMRISyntheticData_test#102.mat %
% showSMsyn_modified(Data_2D_test_template,[1,2],0)


% x_com_reshape = reshape(x_com,[2,4096]);
% P = corrcoef([x_com_reshape',SM']);
% % P: [hat, gt]x[hat, gt]
% P_hat_gt = P(1:2,3:4);
% figure;
% imagesc(abs(P_hat_gt))
% max(abs(P_hat_gt))

% rblist = [0,1,2,8,16,17,21,26];
% bblist = [3,5,11,14,15,19,20,22,24,25,29,31,32,35,36,37];
% rrlist = [4,6,10,13,18,23,27,28,38,41];
% brlist = [7,9,12,30,33,34];

% list = brlist;
% x_com_picked = [];
% m_com_picked = [];
% for i = 1:length(list)
%     loadname = strcat('Testresults#',num2str(list(i)),'.mat');
%     load(loadname)
%     x_com_picked(2*i-1,:,:) = squeeze(x0);
%     x_com_picked(2*i,:,:) = squeeze(x1);
%     m_com_picked(2*i-1,:,:) = squeeze(m0);
%     m_com_picked(2*i,:,:) = squeeze(m1);
% end
% showSMsyn_modified(m_com_picked,[length(list),2],0)
% showSMsyn_modified(x_com_picked,[length(list),2],0)
   

