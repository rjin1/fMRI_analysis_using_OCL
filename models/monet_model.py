"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
from itertools import chain

import torch
from torch import nn, optim
import torch.distributions as tdis

from .base_model import BaseModel
from . import networks

import math
from scipy.io import loadmat
import numpy as np
from util.util import loss_corrcoef

class MONetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=32, lr=1e-4, display_ncols=7, niter_decay=0,
                            dataset_mode='SyntheticFMRI', niter=int(35e8 // 7e4))  # Need to give ncols.
        parser.add_argument('--num_slots', metavar='K', type=int, default=7, help='Number of supported slots')  # Need to give K
        parser.add_argument('--z_dim', type=int, default=16, help='Dimension of individual z latent per slot')  # Need to give z_dim
        if is_train:
            parser.add_argument('--beta', type=float, default=5e-4, help='weight for the encoder KLD')
            parser.add_argument('--epoch_KLDAN', type=int, default=1, help='KLD annealing epoch')
            parser.add_argument('--epoch_KLDAN_start', type=int, default=-1, help='Initial KLD annealing epoch')

            # parser.add_argument('--Mk_l1norm', type=float, default=5e-2, help='The l1 norm coefficient')
            # parser.add_argument('--beta_shapeparam', type=float, default=2, help='KLD Annealing epoch')
            # parser.add_argument('--epoch_Mksparse', type=int, default=1, help='Mask sparsity Annealing epoch')

            parser.add_argument('--GECO_tune', type=bool, default=False, help='GECO tuning')  # Rui
            parser.add_argument('--g_target', type=float, default=12000.0, help='GECO reconstruction target')  # Rui
            parser.add_argument('--g_eta', type=float, default=100, help='GECO step size')  # Rui
            parser.add_argument('--g_alpha', type=float, default=0.99, help='GECO Moving average weight')  # Rui
            
            parser.add_argument('--gamma', type=float, default=5e-1, help='weight for the mask KLD')  # For KLD between mk and sk
            parser.add_argument('--epoch_KLDmkskAN', type=int, default=1, help='KLD annealing epoch')
            parser.add_argument('--epoch_KLDmkskAN_start', type=int, default=-1, help='Initial KLD annealing epoch')
            # parser.add_argument('--tanhalpha', type=float, default=10.0, help='alpha in tanh')
            # parser.add_argument('--gamma_shapeparam', type=float, default=2, help='weight for the mask KLD')

            parser.add_argument('--eta', type=float, default=0, help='slot contrastive loss on latents')
            parser.add_argument('--eta_map', type=float, default=0, help='slot contrastive loss on maps')
            parser.add_argument('--tau_map', type=float, default=0, help='Temperature constant')
            parser.add_argument('--epoch_scl_start', type=int, default=5e4, help='slot contrastive loss start epoch')
            parser.add_argument('--epoch_scl_end', type=int, default=5e4, help='slot contrastive loss stop epoch')

            parser.add_argument('--lambda_corr', type=float, default=0, help='correlation coefficients on components')
            parser.add_argument('--epoch_corr_start', type=int, default=5e4, help='correlation coefficients regularization start epoch')
            parser.add_argument('--epoch_corr_end', type=int, default=5e4, help='correlation coefficients regularization stop epoch')

            parser.add_argument('--lambda_corr_re', type=float, default=0, help='correlation coefficients on components (with reconstruction)')
            parser.add_argument('--epoch_corr_re_start', type=int, default=5e4, help='correlation coefficients regularization start epoch')
            parser.add_argument('--epoch_corr_re_end', type=int, default=5e4, help='correlation coefficients regularization stop epoch')

            parser.add_argument('--lambda_sps', type=float, default=0, help='sparsity regularization coefficient')
            parser.add_argument('--epoch_sps_start', type=int, default=5e4, help='sparsity regularization start epoch')
            parser.add_argument('--epoch_sps_end', type=int, default=5e4, help='sparsity regularizationstop epoch')

            parser.add_argument('--zeta', type=float, default=1, help='weight for the reconstruction')
            parser.add_argument('--epoch_recon_start', type=int, default=0, help='Initial reconstruction annealing epoch')
            # parser.add_argument('--epoch_CELAN', type=int, default=20, help='Cross-entropy loss Annealing epoch')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # self.loss_names = ['E', 'D', 'mask']
        # self.loss_names = ['E', 'D', 'm_k_skewness_sq', 'x_k_masked_norm']
        # self.loss_names = ['E', 'D', 'x_k_masked_norm', 'beta_temp', 'Mkl1norm_temp']
        # self.loss_names = ['E', 'D']
        self.loss_names = ['E', 'D', 'KL_mksk', 'zeta_temp', 'beta_temp', 'gamma_temp']
        if opt.SCL:
            self.loss_names += ['eta_temp', 'slotCEL']
        if opt.SCL_map:
            self.loss_names += ['eta_map_temp', 'slotCEL_map']
        if opt.corr_ica:
            self.loss_names += ['corr']
            self.gica_comps = loadmat(opt.gicacomp_path)[opt.Datatypevariable]
            self.gica_comps = torch.from_numpy(self.gica_comps).float().to(self.device)
            self.singleone = torch.tensor(1).to(self.device)
            self.singlemones = torch.tensor(-1).to(self.device)
        if opt.corr_recon:
            self.loss_names += ['corr_re']
#            self.singleone = torch.tensor(1).to(self.device)
#            self.singlemones = torch.tensor(-1).to(self.device)

        if opt.sparse_map:
            self.loss_names += ['sps']

        # self.loss_names = ['E', 'D', 'x_k_kurtosis']
        # self.loss_names = ['E', 'D', 'beta_temp']
        self.visual_names = ['m{}'.format(i) for i in range(opt.num_slots)] + \
                            ['x{}'.format(i) for i in range(opt.num_slots-1)] + \
                            ['m_tilde{}'.format(i) for i in range(opt.num_slots)] + \
                            ['x_input', 'x_tilde']
                            # ['alpha{}'.format(i) for i in range(opt.num_slots)] + \
                            # ['s{}'.format(i) for i in range(opt.num_slots)] + \
                            # ['xm{}'.format(i) for i in range(opt.num_slots)] + \
        # self.visual_names = ['m{}'.format(i) for i in range(opt.num_slots)] + \
        #                     ['x{}'.format(i) for i in range(opt.num_slots)] + \
        #                     ['x_input', 'x_tilde']
        # self.visual_names = ['x{}'.format(i) for i in range(opt.num_slots)] + \
        #                     ['x', 'x_tilde']
        self.model_names = ['Attn', 'CVAE']
        # self.model_names = ['CVAE']
        self.netAttn = networks.init_net(networks.Attention(opt.input_nc, 1), gpu_ids=self.gpu_ids)
        self.netCVAE = networks.init_net(networks.ComponentVAE(opt.input_nc, opt.z_dim, opt.SCL), gpu_ids=self.gpu_ids)
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.isTrain:  # only defined during training time
            self.criterionKL = nn.KLDivLoss(reduction='batchmean')
            self.optimizer = optim.RMSprop(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=opt.lr)
            # self.optimizer = optim.RMSprop(self.netCVAE.parameters(), lr=opt.lr)
            # self.optimizer = optim.Adam(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=opt.lr)
            self.optimizers = [self.optimizer]
            # Rui:exponential annealing: f(x) = exp(A * (x - B)) + C
            # self.beta_AN_C = opt.beta / (1 - math.exp(opt.epoch_KLDAN * opt.beta_shapeparam))
            # self.beta_AN_B = math.log(-self.beta_AN_C) / (-opt.beta_shapeparam)
            # self.gamma_AN_C = opt.gamma / (1 - math.exp(opt.epoch_KLDmkskAN * opt.gamma_shapeparam))
            # self.gamma_AN_B = math.log(-self.gamma_AN_C) / (-opt.gamma_shapeparam)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.x = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        # self.x_mean = input['A_mean'].to(self.device)
        # self.x_std = input['A_std'].to(self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.loss_E = 0
        self.x_tilde = 0
        if self.opt.corr_ica:
            self.loss_corr = 0

        if self.opt.corr_recon:
            self.loss_corr_re = 0

        if self.opt.sparse_map:
            self.loss_sps = 0

        # Rui: mask threshold:
        # self.thresh_m = 1e-3
        # self.x_sigma = 0
        # b = []
        m = []
        x_mu = []
        # x_mu_scaled_abs = []
        z = []
        z_bilinear = []
        # self.loss_M = 0
        # Rui: Use skewness to regularize masks
        # self.m_k_skewness_sq = 0
        # Rui: Use sparsity to regularize sources
        # self.x_k_masked_norm = 0
        # Rui: Use kurtosis to regularize sources
        # self.x_k_kurtosis = 0

        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(self.x.shape)
        shape[1] = 1
        log_s_k = self.x.new_zeros(shape)
        allones = self.x.new_ones(shape)  # Rui

        for k in range(self.opt.num_slots):
            # Derive mask from current scope
            # if k != self.opt.num_slots - 1:
            if k != self.opt.num_slots - 1:
                log_alpha_k, alpha_logits_k = self.netAttn(self.x, log_s_k)
                # log_alpha_k, alpha_logits_k = self.netAttn(k, shape[0])
                # log_m_k = self.netAttn(k, shape[0])
                log_m_k = log_s_k + log_alpha_k
                # setattr(self, 's{}'.format(k), log_s_k.exp()) # Rui: Check scope
                # setattr(self, 'alpha{}'.format(k), log_alpha_k.exp()) # Rui: Check alpha
                # Compute next scope
                log_s_k += -alpha_logits_k + log_alpha_k
                # m_k = log_m_k.exp()
                # # Rui: Calculate biased skewness:
                # nomi = torch.mean((m_k - m_k.mean(dim=(2, 3), keepdim=True)).pow(3), dim=(2, 3))
                # denomi = m_k.std(dim=(2, 3), unbiased=False).pow(3)
                # m_k_skewness = nomi/denomi
                # self.m_k_skewness_sq += m_k_skewness.pow(2)
                # Get component and mask reconstruction, as well as the z_k parameters
                # x_mu_k, z_mu_k, z_logvar_k, x_mu_k_scaled = self.netCVAE(self.x, log_m_k, self.device, k == 0)
                if self.opt.SCL:
                    x_mu_k, z_mu_k, z_logvar_k, z_k, z_bilinear_k = self.netCVAE(self.x, log_m_k, self.device, k == 0, self.opt.SCL)
                    # Rui: append z_k for slot contrastive loss.
                    z.append(z_k.unsqueeze(dim=1))
                    z_bilinear.append(z_bilinear_k.unsqueeze(dim=1))
                else:
                    x_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0, self.opt.SCL)
                    # xs_coef_k, xs_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)
                    # x_mu_k, z_mu_k, z_logvar_k, z_k, z_bilinear_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)

                # KLD is additive for independent distributions
                self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

                # m_tilde_k = m_tilde_k_logits.exp()
                # x_k_masked = m_tilde_k * x_mu_k
                m_k = log_m_k.exp()
                # x_k_masked = m_k * x_mu_k

                # Rui: scale x_mu_k as probability while maintain its shape.
                # x_mu_k = xs_coef_k.reshape(shape[0], 1, 1, 1) * xs_mu_k
                # x_mu_k_abs = x_mu_k.abs()
                # x_mu_k_abs = torch.tanh(self.opt.tanhalpha * x_mu_k.abs())
                # x_mu_k_abs = x_mu_k.abs() / x_mu_k.abs().amax(dim=(2, 3), keepdim=True)
                # x_mu_k_abs = x_mu_k.abs()
                # # Rui: Calculate biased skewness:
                # nomi = torch.mean((x_k_masked - x_k_masked.mean(dim=(2, 3), keepdim=True)).pow(3), dim=(2, 3))
                # denomi = x_k_masked.std(dim=(2, 3), unbiased=False).pow(3)
                # m_k_skewness = nomi/denomi
                # self.m_k_skewness_sq += m_k_skewness.pow(2)

                # Rui: Calculate l1-norm of maps
                # if self.opt.sparse_map & (epoch_train >= self.opt.epoch_sps_start) & (
                        # epoch_train <= self.opt.epoch_sps_end):
                    # self.loss_sps += torch.norm(x_mu_k, p=1, dim=[2, 3, 4])
                    # self.sps_bp = 1
                # else:
                    # self.sps_bp = 0

                # self.x_k_masked_norm += torch.norm(x_mu_k, p=1, dim=[2, 3])
                # m_k_temp = m_k / torch.norm(m_k, p=2, dim=(2, 3), keepdim=True)
                # x_mu_k_temp = x_mu_k / torch.norm(x_mu_k, p=2, dim=(2, 3), keepdim=True)
                # self.x_k_masked_norm += (m_k_temp.pow(2) + x_mu_k_temp.pow(2)).sqrt().sum()
                # self.x_k_masked_norm += (m_k.pow(2) + x_mu_k.pow(2)).sqrt().sum()
                # self.x_k_masked_norm += torch.norm(m_k, p=1, dim=(2, 3))


                # # Rui: Calculate biased Kurtosis:
                # nomi = torch.mean((x_mu_k - x_mu_k.mean(dim=(2, 3), keepdim=True)).pow(4), dim=(2, 3))
                # denomi = x_mu_k.std(dim=(2, 3), unbiased=False).pow(4)
                # x_k_kurtosis_temp = nomi/denomi
                # self.x_k_kurtosis += x_k_kurtosis_temp

                # # Rui: Calculate biased Kurtosis:
                # nomi = torch.mean((x_k_masked - x_k_masked.mean(dim=(2, 3), keepdim=True)).pow(4), dim=(2, 3))
                # denomi = x_k_masked.std(dim=(2, 3), unbiased=False).pow(4)
                # x_k_kurtosis_temp = nomi / denomi
                # self.x_k_kurtosis += x_k_kurtosis_temp

                # Rui: Correlation coefficient regularization
                # if self.opt.corr_ica & (epoch_train >= self.opt.epoch_corr_start) & (epoch_train <= self.opt.epoch_corr_end):
                    # # Only for fmri data with N_batch x 1 comp x 3D size
                    # self.loss_corr += loss_corrcoef(x_mu_k, self.gica_comps.index_select(0, torch.tensor(k).to(self.device)), self.singleone, self.singlemones).sum()
                    # self.corr_ica_bp = 1
                # else:
                    # self.corr_ica_bp = 0

                # Iteratively reconstruct the output image
                # self.x_tilde += x_k_masked
                # self.x_sigma += x_signam_k
                # self.loss_M += torch.norm(m_k, p=1)
                self.x_tilde += x_mu_k
                # Rui: threshold m_k, then do dot product
                # m_k_thres = (m_k > self.thresh_m).float()
                # self.x_tilde += x_mu_k * m_k_thres
                # Accumulate
                # m.append(m_k)
                m.append(log_m_k)
                x_mu.append(x_mu_k)
                # x_mu_scaled_abs.append(x_mu_k_scaled.abs())
                setattr(self, 'x{}'.format(k), x_mu_k)
                # setattr(self, 'xs{}'.format(k), xs_mu_k)
            else:
                log_m_k = log_s_k
                # setattr(self, 's{}'.format(k), log_s_k.exp()) # Rui: Check scope
                # setattr(self, 'alpha{}'.format(k), log_alpha_k.exp()) # Rui: Check alpha
                # log_m_k = self.netAttn(k, shape[0])
                m_k = log_m_k.exp()
                #
                # x_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)
                # # KLD is additive for independent distributions
                # self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()
                # self.x_tilde += x_mu_k
                #
                # # m.append(m_k)
                m.append(log_m_k)
                # setattr(self, 'x{}'.format(k), x_mu_k)

            # Get component and mask reconstruction, as well as the z_k parameters
            # m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)
            # x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)
            # x_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, self.device, k == 0)
            # x_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_s_k, k == 0)
            # x_mu_k, x_signam_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, k == 0)
            # m_tilde_k_logits, x_mu_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, k == 0)

            # KLD is additive for independent distributions
            # self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

            # m_tilde_k = m_tilde_k_logits.exp()
            # x_k_masked = m_tilde_k * x_mu_k
            # m_k = log_m_k.exp()
            # x_k_masked = m_k * x_mu_k

            # # Rui: Calculate biased skewness:
            # nomi = torch.mean((x_k_masked - x_k_masked.mean(dim=(2, 3), keepdim=True)).pow(3), dim=(2, 3))
            # denomi = x_k_masked.std(dim=(2, 3), unbiased=False).pow(3)
            # m_k_skewness = nomi/denomi
            # self.m_k_skewness_sq += m_k_skewness.pow(2)

            # Rui: Calculate l1-norm of sources
            # self.x_k_masked_norm += torch.norm(x_mu_k, p=1, dim=(2, 3))

            # # Rui: Calculate biased Kurtosis:
            # nomi = torch.mean((x_mu_k - x_mu_k.mean(dim=(2, 3), keepdim=True)).pow(4), dim=(2, 3))
            # denomi = x_mu_k.std(dim=(2, 3), unbiased=False).pow(4)
            # x_k_kurtosis_temp = nomi/denomi
            # self.x_k_kurtosis += x_k_kurtosis_temp

            # # Rui: Calculate biased Kurtosis:
            # nomi = torch.mean((x_k_masked - x_k_masked.mean(dim=(2, 3), keepdim=True)).pow(4), dim=(2, 3))
            # denomi = x_k_masked.std(dim=(2, 3), unbiased=False).pow(4)
            # x_k_kurtosis_temp = nomi / denomi
            # self.x_k_kurtosis += x_k_kurtosis_temp

            # Exponents for the decoder loss
            # b_k = log_m_k - 0.5 * x_logvar_k[0] - (self.x * 0.5 + 0.5 - x_mu_k).pow(2) / (2 * x_logvar_k[0].exp())
            # b.append(b_k.unsqueeze(1))

            # Iteratively reconstruct the output image
            # self.x_tilde += x_k_masked
            # self.x_sigma += x_signam_k
            # self.loss_M += torch.norm(m_k, p=1)
            # self.x_tilde += x_mu_k

            # Get outputs for kth step
            # setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
            setattr(self, 'm{}'.format(k), m_k)        
            # setattr(self, 'm{}'.format(k), m_tilde_k)
            # setattr(self, 'xm{}'.format(k), x_k_masked)
            # setattr(self, 'x_input', self.x * 0.5 + 0.5)
            setattr(self, 'x_input', self.x)
            # setattr(self, 'x_input', self.x * self.x_std + self.x_mean)
            setattr(self, 'x_tilde', self.x_tilde)
            # Accumulate
            # m.append(m_k)
            # m_tilde_logits.append(m_tilde_k_logits)

        # self.b = - 0.5 * x_logvar_k - (self.x - self.x_tilde).pow(2) / (2 * x_logvar_k.exp())
        # Rui: x*0.5 + 0.5 = x_input
        # self.b = - (2 * torch.sqrt(torch.tensor(0.5))).log() - (torch.abs((self.x * 0.5 + 0.5 - self.x_tilde)) / (torch.sqrt(torch.tensor(0.5))))
        self.b = - (2 * torch.sqrt(torch.tensor(0.5))).log() - (torch.abs((self.x - self.x_tilde)) / (torch.sqrt(torch.tensor(0.5))))
        # self.b = - (2 * torch.sqrt(torch.tensor(0.5))).log() - (
        #             torch.abs((self.x - ((self.x_tilde - self.x_mean) / self.x_std))) / (torch.sqrt(torch.tensor(0.5))))
        # self.b = tdis.normal.Normal(self.x_tilde, self.x_sigma).log_prob(self.x)
        # self.b = tdis.laplace.Laplace(self.x_tilde, torch.sqrt(torch.tensor(0.5)).to(self.device)).log_prob(self.x)
        # b.append(b_k.unsqueeze(1))
        # self.b = torch.cat(b, dim=1)
        self.m = torch.cat(m, dim=1)
        self.x_mu = torch.cat(x_mu, dim=1)
        # self.m_tilde = torch.tanh(self.opt.tanhalpha * self.x_mu.abs())
        self.m_tilde = self.x_mu.abs()

        # Rui: scale m_tilde to make the sum over k less than 1.
        self.m_tilde = self.m_tilde / self.m_tilde.sum(dim=1, keepdim=True).amax(dim=(2, 3, 4), keepdim=True)
        # Rui: Add some epsilon to improve stability
        # m_tilde_bg = allones + 1e-5 - self.m_tilde.sum(dim=1, keepdim=True)
        m_tilde_bg = allones * self.m_tilde.sum(dim=1, keepdim=True).amax(dim=(2, 3, 4), keepdim=True) + 1e-5 - self.m_tilde.sum(dim=1, keepdim=True)
        self.m_tilde = torch.cat((self.m_tilde, m_tilde_bg), dim=1)

        # Rui save these masks
        m_tilde_tosave = self.m_tilde.split(1, dim=1)
        for k in range(self.opt.num_slots):
            setattr(self, 'm_tilde{}'.format(k), m_tilde_tosave[k])

        # # Rui: calculate slot contrastive loss:
        # self.x_mu_abs = torch.cat(x_mu_abs, dim=1)
        # # self.x_mu_abs = torch.cat(x_mu_scaled_abs, dim=1)
        # self.x_mu_abs = self.x_mu_abs.view(shape[0], self.opt.num_slots - 1, shape[2] * shape[3])
        # self.x_mu_logits = torch.matmul(self.x_mu_abs, self.x_mu_abs.transpose(2, 1))
        # self.inp = self.x_mu_logits.reshape(shape[0] * (self.opt.num_slots - 1), -1)
        # if self.opt.SCL:
        #    self.z = torch.cat(z, dim=1)
        #    self.z_bilinear = torch.cat(z_bilinear, dim=1)
        #    self.z_logits = torch.matmul(self.z_bilinear, self.z.transpose(2, 1).roll(1, 0))
        #    self.n_pairs = self.z_logits.shape[0]
        #    self.inp = self.z_logits.reshape(self.n_pairs * (self.opt.num_slots-1), -1)
        #    self.target = torch.cat([torch.arange(self.opt.num_slots-1) for i in range(self.n_pairs)]).to(self.device)
        # # test_debug_bp = 0 
        # if self.opt.SCL_map & (epoch_train >= self.opt.epoch_scl_start) & (epoch_train <= self.opt.epoch_scl_end):
            # self.x_mu_re = self.x_mu.view(shape[0], self.opt.num_slots-1, shape[2] * shape[3] * shape[4])
            # self.x_mu_norm = nn.functional.normalize(self.x_mu_re, dim=2)
            # self.x_mu_logits = torch.matmul(self.x_mu_norm, self.x_mu_norm.transpose(2, 1).roll(1, 0)) / self.opt.tau_map
            # self.n_pairs = self.x_mu_logits.shape[0]
            # self.x_mu_inp = self.x_mu_logits.reshape(self.n_pairs * (self.opt.num_slots - 1), -1)
            # self.x_mu_target = torch.cat([torch.arange(self.opt.num_slots - 1) for i in range(self.n_pairs)]).to(self.device)
            # self.SCL_map_bp = 1
        # else:
            # self.SCL_map_bp = 0

        # Cross-correlation between maps and reconstruction
        # if self.opt.corr_recon & (epoch_train >= self.opt.epoch_corr_re_start) & (
                # epoch_train <= self.opt.epoch_corr_re_end):
            # # Only for fmri data with N_batch x 1 comp x 3D size
            # self.x_tilde_re = self.x_tilde.view(shape[0], shape[1], shape[2] * shape[3] * shape[4])
            # self.x_mu_re = self.x_mu.view(shape[0], self.opt.num_slots - 1, shape[2] * shape[3] * shape[4])
            # self.x_tilde_norm = nn.functional.normalize(self.x_tilde_re, dim=2)
            # self.x_mu_norm = nn.functional.normalize(self.x_mu_re, dim=2)

            # self.x_tilde_norm = self.x_tilde_norm.repeat(1, self.opt.num_slots - 1, 1)

            # self.x_mu_tilde_croscorr = self.x_mu_norm * self.x_tilde_norm

            # self.loss_corr_re = self.x_mu_tilde_croscorr.sum()
            # self.corr_recon_bp = 1
        # else:
            # self.corr_recon_bp = 0

    def backward(self, epoch_train):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_beta_temp = min(self.opt.beta * (max((epoch_train - self.opt.epoch_KLDAN_start), 0) / self.opt.epoch_KLDAN), self.opt.beta)
        # self.loss_beta_temp = self.opt.beta
        # self.loss_beta_temp = min(math.exp(self.opt.beta_shapeparam * (epoch_train - self.beta_AN_B)) + self.beta_AN_C, self.opt.beta)
        # self.loss_Mkl1norm_temp = min(self.opt.Mk_l1norm * (epoch_train / self.opt.epoch_Mksparse), self.opt.Mk_l1norm)
        # self.loss_Mkl1norm_temp = self.opt.Mk_l1norm
        self.loss_gamma_temp = min(self.opt.gamma * (max((epoch_train - self.opt.epoch_KLDmkskAN_start), 0) / self.opt.epoch_KLDmkskAN), self.opt.gamma)
        # self.loss_gamma_temp = min(math.exp(self.opt.gamma_shapeparam * (epoch_train - self.gamma_AN_B)) + self.gamma_AN_C, self.opt.gamma)
        # self.loss_gamma_temp = self.opt.gamma
        self.loss_zeta_temp = self.opt.zeta if epoch_train >= self.opt.epoch_recon_start else 0.0
        n = self.x.shape[0]
        self.loss_E /= n
        # self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
        self.loss_D = -self.b.sum() / n
        # self.loss_m_k_skewness_sq = self.m_k_skewness_sq.sum() / n
        # self.loss_sk_norm = self.x_k_masked_norm.sum() / n
        # self.loss_x_k_masked_norm = self.x_k_masked_norm / n
        # self.loss_x_k_kurtosis = -self.x_k_kurtosis.sum() / n
        # self.loss_KL_mksk = self.criterionKL(self.m.log(), self.m_tilde)
        self.loss_KL_mksk = self.criterionKL(self.m, self.m_tilde.detach())
        # self.loss_KL_mksk = self.criterionKL(self.m, self.m_tilde)
        # self.loss_KL_mksk = self.criterionKL(self.m.log(), self.m_tilde.detach())
        loss = self.loss_zeta_temp * self.loss_D + self.loss_beta_temp * self.loss_E + self.loss_gamma_temp * self.loss_KL_mksk
        # Rui: slot contrastive loss
        # if self.opt.SCL:
            # self.loss_eta_temp = min(self.opt.eta * (epoch_train / self.opt.epoch_CELAN), self.opt.eta)
        #    self.loss_eta_temp = self.opt.eta if epoch_train >= self.opt.epoch_scl_start else 0.0
        #    self.loss_eta_temp = self.opt.eta if epoch_train <= self.opt.epoch_scl_end else 0.0

        #    self.loss_slotCEL = nn.CrossEntropyLoss(reduction='none')(self.inp, self.target).sum() / self.n_pairs
        #    loss += self.loss_eta_temp * self.loss_slotCEL

        # if self.SCL_map_bp:
            # # self.loss_eta_temp = min(self.opt.eta * (epoch_train / self.opt.epoch_CELAN), self.opt.eta)
            # self.loss_eta_map_temp = self.opt.eta_map
            # self.loss_slotCEL_map = nn.CrossEntropyLoss(reduction='none')(self.x_mu_inp, self.x_mu_target).sum() / self.n_pairs            
        # else:
            # self.loss_eta_map_temp = 0
            # self.loss_slotCEL_map = 0
            
        # loss += self.loss_eta_map_temp * self.loss_slotCEL_map
                
        # if self.corr_ica_bp:
            # self.loss_lambdacorr_temp = self.opt.lambda_corr 
            # self.loss_corr /= n
        # else:        
            # self.loss_lambdacorr_temp = 0
            # self.loss_corr = 0
            
        # loss += self.loss_lambdacorr_temp * self.loss_corr

        # if self.corr_recon_bp:
            # self.loss_lambdacorr_re_temp = self.opt.lambda_corr_re
            # self.loss_corr_re /= n
        # else:
            # self.loss_lambdacorr_re_temp = 0
            # self.loss_corr_re = 0

        # loss += self.loss_lambdacorr_re_temp * self.loss_corr_re

        # if self.sps_bp:
            # self.loss_lambda_sps_temp = self.opt.lambda_sps
            # self.loss_sps = self.loss_sps.sum() / n
        # else:
            # self.loss_lambda_sps_temp = 0
            # self.loss_sps = 0

        # loss += self.loss_lambda_sps_temp * self.loss_sps

        # loss = self.loss_D + self.opt.beta * self.loss_E + self.opt.gamma * self.loss_mask
        # loss = self.loss_D + self.opt.beta * self.loss_E + 1e5 * self.loss_m_k_skewness_sq + 1e-3 * self.loss_x_k_masked_norm
        # loss = self.loss_D + self.opt.beta * self.loss_E + 2.9e-3 * self.loss_x_k_kurtosis
        # loss = self.loss_D + self.opt.beta * self.loss_E + 5e-5 * self.loss_x_k_masked_norm
        # loss = self.loss_D + self.opt.beta * self.loss_E
        # loss = self.loss_D + self.loss_beta_temp * self.loss_E + self.loss_gamma_temp * self.loss_KL_mksk + self.opt.Mk_l1norm * self.loss_sk_norm
        # loss = self.loss_D + self.loss_beta_temp * self.loss_E
        # loss = self.loss_D + self.loss_beta_temp * self.loss_E + self.loss_gamma_temp * self.loss_KL_mksk
        # loss = self.loss_D + self.loss_beta_temp * self.loss_E + self.loss_gamma_temp * self.loss_KL_mksk
        loss.backward()

    def optimize_parameters(self, epoch_train):
        """Update network weights; it will be called in every training iteration."""
        self.forward()    # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward(epoch_train)   # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
