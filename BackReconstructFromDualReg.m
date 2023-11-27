function [tchat, shat] = BackReconstructFromDualReg(icasig, data_fmri)
    icasig = icasig';
    icasig_demean = icasig - repmat(mean(icasig), [size(icasig, 1), 1]);

    data_fmri_demean = data_fmri - repmat(mean(data_fmri), [size(data_fmri, 1), 1]);

    tchat = pinv(icasig_demean) * data_fmri_demean;
    tchat = tchat';

    tchat_demean = tchat - repmat(mean(tchat), [size(tchat, 1), 1]);
    data_fmri_t = data_fmri';
    data_fmri_t_demean = data_fmri_t - repmat(mean(data_fmri_t), [size(data_fmri_t, 1), 1]);
    shat = pinv(tchat_demean) * data_fmri_t_demean;
end