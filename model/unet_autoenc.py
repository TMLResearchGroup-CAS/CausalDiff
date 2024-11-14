from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *

from .wide_resnet import BasicBlock
from .wide_resnet import NetworkBlock
from torch import nn, optim, autograd
import numpy as np

# std = 1.

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
	def __init__(self, type = '3d'):
		super(UnFlatten, self).__init__()
		self.type = type
	def forward(self, input):
		if self.type == '3d':
			return input.view(input.size(0), input.size(1), 1, 1, 1)
		else:
			return input.view(input.size(0), input.size(1), 1, 1)


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
	# number of style channels
	enc_out_channels: int = 512
	enc_attn_resolutions: Tuple[int] = None
	enc_pool: str = 'depthconv'
	enc_num_res_block: int = 2
	enc_channel_mult: Tuple[int] = None
	enc_grad_checkpoint: bool = False
	latent_net_conf: MLPSkipNetConfig = None
	club_hidden_dim: int = 64
	use_club: bool = False
	consistency: bool = False
	mask_threshold: float = None

	def make_model(self):
		return BeatGANsAutoencModel(self)


class d_LaCIM(nn.Module):
	def __init__(self,
					in_channel=1,
					zs_dim=256,
					num_classes=1,
					decoder_type=0,
					total_env=2,
					is_cuda=1
					):
		
		super(d_LaCIM, self).__init__()
		# print('model: d_LaCIM, zs_dim: %d' % zs_dim)
		self.in_channel = in_channel
		self.num_classes = num_classes
		self.zs_dim = zs_dim
		self.decoder_type = decoder_type
		self.total_env = total_env
		self.is_cuda = is_cuda
		self.in_plane = zs_dim
		self.z_dim = int(round(zs_dim * 0.5))
		self.Enc_x = self.get_Enc_x_28()
		self.u_dim = total_env
		print('z_dim is ', self.z_dim)
		self.s_dim = int(self.zs_dim - self.z_dim)
		self.mean_z = []
		self.logvar_z = []
		self.mean_s = []
		self.logvar_s = []

		self.shared_s1 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_s2 = nn.Linear(self.in_plane, self.s_dim)
		self.shared_s3 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_s4 = nn.Linear(self.in_plane, self.s_dim)
		self.shared_z1 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_z2 = nn.Linear(self.in_plane, self.z_dim)
		self.shared_z3 = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
		self.shared_z4 = nn.Linear(self.in_plane, self.z_dim)

		for env_idx in range(self.total_env):
			self.mean_z.append(
				nn.Sequential(
					# self.Fc_bn_ReLU(self.in_plane, self.in_plane),
					# nn.Linear(self.in_plane, self.z_dim)
					self.shared_z1,
					self.shared_z2
				)
			)
			self.logvar_z.append(
				nn.Sequential(
					# self.Fc_bn_ReLU(self.in_plane, self.in_plane),
					# nn.Linear(self.in_plane, self.z_dim)
					self.shared_z3,
					self.shared_z4
				)
			)
			self.mean_s.append(
				nn.Sequential(
					# self.shared_s,
					# nn.Linear(self.in_plane, self.s_dim)
					self.shared_s1,
					self.shared_s2
				)
			)
			self.logvar_s.append(
				nn.Sequential(
					# self.shared_s,
					# nn.Linear(self.in_plane, self.s_dim)
					self.shared_s3,
					self.shared_s4
				)
			)

		self.mean_z = nn.ModuleList(self.mean_z)
		self.logvar_z = nn.ModuleList(self.logvar_z)
		self.mean_s = nn.ModuleList(self.mean_s)
		self.logvar_s = nn.ModuleList(self.logvar_s)
		
		# prior
		self.Enc_u_prior = self.get_Enc_u()
		self.mean_zs_prior = nn.Sequential(
			nn.Linear(32, self.zs_dim))
		self.logvar_zs_prior = nn.Sequential(
			nn.Linear(32, self.zs_dim))
		
		self.Dec_y = self.get_Dec_y()
		self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))


	def get_zs(self, x, target, env, adv_mode='none'):
		x = self.Enc_x(x)
		mu, logvar = self.encode(x, env)
		zs = self.reparametrize(mu, logvar)
		z = zs[:, :self.z_dim]
		s = zs[:, self.z_dim:]
		# adversarial evaluate mode
		# if adv_mode == 'z':
		# 	z = z+self.delta
		# elif adv_mode == 's':
		# 	s = s+self.delta
		return z, s

	def get_pred_y(self, x, env):
		x = self.Enc_x(x)
		mu, logvar = self.encode(x, env)
		zs = self.reparametrize(mu, logvar)
		z = zs[:, :self.z_dim]
		s = zs[:, self.z_dim:]
		zs = torch.cat([z, s], dim=1)
		pred_y = self.Dec_y(zs[:, self.z_dim:])
		return pred_y

	def get_x_y(self, z, s):
		zs = torch.cat([z, s], dim=1)
		rec_x = self.Dec_x(zs)
		pred_y = self.Dec_y(zs[:, self.z_dim:])
		return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

	def get_y(self, s):
		return self.Dec_y(s)

	def get_y_by_zs(self, mu, logvar, env):
		zs = self.reparametrize(mu, logvar)
		s = zs[:, self.z_dim:]
		return self.Dec_y(s)

	def encode_mu_var(self, x, env_idx=0):
		return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)] ,dim=1), \
				torch.cat([self.logvar_z[env_idx](x), self.logvar_s[env_idx](x)], dim=1)

	def encode_prior(self, x, env_idx):
		temp = env_idx * torch.ones(x.size()[0], 1)
		temp = temp.long().to(x.device)#.cuda()
		y_onehot = torch.FloatTensor(x.size()[0], self.total_env).to(x.device)#.cuda()
		y_onehot.zero_()
		y_onehot.scatter_(1, temp, 1)
		# print(env_idx, y_onehot, 'onehot')
		u = self.Enc_u_prior(y_onehot)
		#return self.mean_zs_prior(u), self.logvar_zs_prior(u)
		default_s = torch.randn(x.size(0), self.s_dim).to(x.device)#.cuda()
		return torch.cat([self.mean_zs_prior(u)[:, :self.z_dim], default_s], dim=1), \
				torch.cat([self.logvar_zs_prior(u)[:, :self.z_dim], default_s], dim=1)

	def decode_x(self, zs):
		return self.Dec_x(zs)

	def decode_y(self, s):
		return self.Dec_y(s)

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		if self.is_cuda:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		# eps = torch.FloatTensor(std.size()).normal_()
		return eps.mul(std).add_(mu)


	def get_Dec_y(self):
		return nn.Sequential(
			# self.Fc_bn_ReLU(int(self.zs_dim), 512),
			self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
			self.Fc_bn_ReLU(512, 256),
			nn.Linear(256, self.num_classes),
			nn.Softmax(dim=1),
		)

	def get_Enc_u(self):
		return nn.Sequential(
			self.Fc_bn_ReLU(self.u_dim, 16),
			self.Fc_bn_ReLU(16, 32)
		)

	def get_Enc_x_28(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
		# nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
		nChannels = [16, 16 * widen_factor, 32 * widen_factor, self.zs_dim]
		assert ((depth - 4) % 6 == 0)
		n = (depth - 4) / 6
		block = BasicBlock
		# 1st conv before any network block
		self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
							padding=1, bias=False)
		# 1st block
		self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
		if sub_block1:
			# 1st sub-block
			self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
		# 2nd block
		self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
		# 3rd block
		self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
		# global average pooling and classifier
		self.bn1 = nn.BatchNorm2d(nChannels[3])
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
		self.nChannels = nChannels[3]

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear) and not m.bias is None:
				m.bias.data.zero_()
		return nn.Sequential(
			self.conv1,
			self.block1,
			self.block2,
			self.block3,
			self.bn1,
			self.relu,
			# F.avg_pool2d(out, 8),
		# 	self.Conv_bn_ReLU(self.in_channel, 32),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(32, 64),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(64, 128),
		# 	nn.MaxPool2d(2),
		# 	self.Conv_bn_ReLU(128, 256),
			nn.AdaptiveAvgPool2d(1),
			Flatten(),
		)

	def Fc_bn_ReLU(self, in_channels, out_channels):
		layer = nn.Sequential(
			nn.Linear(in_channels, out_channels),
			nn.BatchNorm1d(out_channels),
			nn.ReLU())
		return layer

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
	'''
		This class provides the CLUB estimation to I(X,Y)
		Method:
			forward() :      provides the estimation with input samples  
			loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
		Arguments:
			x_dim, y_dim :         the dimensions of samples from X, Y respectively
			hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
			x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
	'''
	def __init__(self, x_dim, y_dim, hidden_size):
		super(CLUB, self).__init__()
		# p_mu outputs mean of q(Y|X)
		#print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
		self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									nn.ReLU(),
									nn.Linear(hidden_size//2, y_dim))
		# p_logvar outputs log of variance of q(Y|X)
		self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
									nn.ReLU(),
									nn.Linear(hidden_size//2, y_dim),
									nn.Tanh())

	def get_mu_logvar(self, x_samples):
		mu = self.p_mu(x_samples)
		logvar = self.p_logvar(x_samples)
		return mu, logvar
	
	def forward(self, x_samples, y_samples): 
		mu, logvar = self.get_mu_logvar(x_samples)
		
		# log of conditional probability of positive sample pairs
		positive = - (mu - y_samples)**2 /2./logvar.exp()  
		
		prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
		y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

		# log of conditional probability of negative sample pairs
		negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

		return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

	def loglikeli(self, x_samples, y_samples, index_mask=None): # unnormalized loglikelihood 
		mu, logvar = self.get_mu_logvar(x_samples)
		ll = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1)
		if index_mask is not None:
			ll[index_mask] = 0.
		return ll.mean(dim=0)
	
	def learning_loss(self, x_samples, y_samples, index_mask=None):
		return - self.loglikeli(x_samples, y_samples, index_mask)

class BeatGANsAutoencModel(BeatGANsUNetModel):
	def __init__(self, conf: BeatGANsAutoencConfig):
		super().__init__(conf)
		self.conf = conf

		# print('conf: ')
		# print(conf.model_channels)
		# print(conf.embed_channels)
		# having only time, cond
		self.time_embed = TimeStyleSeperateEmbed(
			time_channels=conf.model_channels,
			time_out_channels=conf.embed_channels,
		)
		# self.time_embed = nn.Sequential(
		# 	linear(self.time_emb_channels, conf.embed_channels),
		# 	nn.SiLU(),
		# 	linear(conf.embed_channels, conf.embed_channels),
		# )

		# self.encoder = BeatGANsEncoderConfig(
		# 	image_size=conf.image_size,
		# 	in_channels=conf.in_channels,
		# 	model_channels=conf.model_channels,
		# 	out_hid_channels=conf.enc_out_channels,
		# 	out_channels=conf.enc_out_channels,
		# 	num_res_blocks=conf.enc_num_res_block,
		# 	attention_resolutions=(conf.enc_attn_resolutions
		# 						or conf.attention_resolutions),
		# 	dropout=conf.dropout,
		# 	channel_mult=conf.enc_channel_mult or conf.channel_mult,
		# 	use_time_condition=False,
		# 	conv_resample=conf.conv_resample,
		# 	dims=conf.dims,
		# 	use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
		# 	num_heads=conf.num_heads,
		# 	num_head_channels=conf.num_head_channels,
		# 	resblock_updown=conf.resblock_updown,
		# 	use_new_attention_order=conf.use_new_attention_order,
		# 	pool=conf.enc_pool,
		# ).make_model()

		if conf.latent_net_conf is not None:
			self.latent_net = conf.latent_net_conf.make_model()
		
		self.lacim = d_LaCIM(in_channel=3,
							zs_dim=self.conf.enc_out_channels,
							num_classes=10,
							decoder_type=1,
							total_env=1,)
		# self.conf.use_club = True
		if self.conf.use_club:
			# print('club built!!!')
			self.club = CLUB(x_dim=self.lacim.z_dim, 
							y_dim=self.lacim.s_dim,
							hidden_size=self.conf.club_hidden_dim,)


	# def encode_mu_var(self, x, env_idx=0):
	# 	# print(x.size())
	# 	# print(self.mean_z[env_idx](x).size())
	# 	# print(self.mean_s[env_idx](x).size())
	# 	# print(self.logvar_z[env_idx](x).size())
	# 	# print(self.logvar_s[env_idx](x).size())
	# 	return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)] ,dim=1), \
	# 	torch.cat([self.logvar_z[env_idx](x), self.logvar_s[env_idx](x)], dim=1)

	# def encode_prior(self, x, env_idx=0):
	# 	temp = env_idx * torch.ones(x.size()[0], 1)
	# 	temp = temp.long().to(x.device)#.cuda()
	# 	# y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).to(x.device)#.cuda()
	# 	y_onehot = torch.FloatTensor(x.size()[0], self.total_env).to(x.device)#.cuda()
	# 	y_onehot.zero_()
	# 	y_onehot.scatter_(1, temp, 1)
	# 	# print(env_idx, y_onehot, 'onehot')
	# 	u = self.Enc_u_prior(y_onehot)
	# 	#return self.mean_zs_prior(u), self.logvar_zs_prior(u)
	# 	default_s = torch.randn(x.size(0), self.s_dim).to(x.device)#.cuda()
	# 	return torch.cat([self.mean_zs_prior(u)[:, :self.z_dim], default_s], dim=1), \
	# 			torch.cat([self.logvar_zs_prior(u)[:, :self.z_dim], default_s], dim=1)

	
	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.cuda.FloatTensor(std.size()).normal_()
		# eps = torch.FloatTensor(std.size()).normal_()
		return eps.mul(std).add_(mu)

	def sample_z(self, n: int, device):
		assert self.conf.is_stochastic
		return torch.randn(n, self.conf.enc_out_channels, device=device)

	def noise_to_cond(self, noise: Tensor):
		raise NotImplementedError()
		assert self.conf.noise_net_conf is not None
		return self.noise_net.forward(noise)

	def encode(self, x):
		# if r is None:
		# 	r = torch.zeros(x.size(0))
		x = self.lacim.Enc_x(x)
		mu, logvar = self.lacim.encode_mu_var(x, env_idx=0)
		mu_prior, logvar_prior = self.lacim.encode_prior(x, env_idx=0)
		# mu, logvar = [self.lacim.encode_mu_var(x, env_idx=i) for i in range(self.lacim.total_env)]
		# mu = torch.stack([mu[env_idx][i] for i, env_idx in enumerate(r)])
		# logvar = torch.stack([logvar[env_idx][i] for i, env_idx in enumerate(r)])

		# mu_prior, logvar_prior = [self.lacim.encode_prior(x, env_idx=i) for i in range(self.lacim.total_env)]
		# mu_prior = torch.stack([mu_prior[env_idx][i] for i, env_idx in enumerate(r)])
		# logvar_prior = torch.stack([logvar_prior[env_idx][i] for i, env_idx in enumerate(r)])

		cond = self.reparameterize(mu, logvar)
		# z = zs[:, :self.z_dim]
		# s = zs[:, self.z_dim:]
		# pred_y = self.Dec_y(zs)
		return {'cond': cond, 'mu':mu, 'logvar':logvar, 'mu_prior':mu_prior, 'logvar_prior':logvar_prior}
	
	def dec_y(self, s):
		return self.lacim.Dec_y(s)

	@property
	def stylespace_sizes(self):
		modules = list(self.input_blocks.modules()) + list(
			self.middle_block.modules()) + list(self.output_blocks.modules())
		sizes = []
		for module in modules:
			if isinstance(module, ResBlock):
				linear = module.cond_emb_layers[-1]
				sizes.append(linear.weight.shape[0])
		return sizes

	def encode_stylespace(self, x, return_vector: bool = True):
		"""
		encode to style space
		"""
		modules = list(self.input_blocks.modules()) + list(
			self.middle_block.modules()) + list(self.output_blocks.modules())
		# (n, c)
		cond = self.encoder.forward(x)
		S = []
		for module in modules:
			if isinstance(module, ResBlock):
				# (n, c')
				s = module.cond_emb_layers.forward(cond)
				S.append(s)

		if return_vector:
			# (n, sum_c)
			return torch.cat(S, dim=1)
		else:
			return S

	def forward(self,
				x,
				t,
				y=None,
				x_start=None,
				cond=None,
				style=None,
				noise=None,
				t_cond=None,
				**kwargs):
		"""
		Apply the model to an input batch.

		Args:
			x_start: the original image to encode
			cond: output of the encoder
			noise: random noise (to predict the cond)
		"""

		if t_cond is None:
			t_cond = t

		if noise is not None:
			# if the noise is given, we predict the cond from noise
			cond = self.noise_to_cond(noise)

		if cond is None:
			mode = 'train'
			if x is not None:
				assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

			tmp = self.encode(x_start)
			cond = tmp['cond']

		if t is not None:
			_t_emb = timestep_embedding(t, self.conf.model_channels)
			_t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
		else:
			# this happens when training only autoenc
			_t_emb = None
			_t_cond_emb = None

		if self.conf.resnet_two_cond:
			res = self.time_embed.forward(
				time_emb=_t_emb,
				cond=cond,
				time_cond_emb=_t_cond_emb,
			)
		else:
			raise NotImplementedError()

		if self.conf.resnet_two_cond:
			# two cond: first = time emb, second = cond_emb
			emb = res.time_emb
			cond_emb = res.emb
		else:
			# one cond = combined of both time and cond
			emb = res.emb
			cond_emb = None

		# if cond.requires_grad:
		# 	print('cond_emb grad: ')
		# 	print(autograd.grad(cond_emb, cond, torch.ones_like(cond_emb)))

		# override the style if given
		style = style or res.style

		assert (y is not None) == (
			self.conf.num_classes is not None
		), "must specify y if and only if the model is class-conditional"

		if self.conf.num_classes is not None:
			raise NotImplementedError()
			# assert y.shape == (x.shape[0], )
			# emb = emb + self.label_emb(y)

		# where in the model to supply time conditions
		enc_time_emb = emb
		mid_time_emb = emb
		dec_time_emb = emb
		# where in the model to supply style conditions
		# print('cond in unet_autoenc : ')
		# cond_emb = None
		# print(cond_emb)

		# cond_emb_gen = cond_emb.clone()
		# if mode == 'train' and self.conf.mask_threshold is not None:
		# 	index_mask = np.where(np.random.rand(x_start.size(0)) < self.conf.mask_threshold)
		# 	cond_emb_gen[index_mask] = 0

		# cond_emb = None
		enc_cond_emb = cond_emb
		mid_cond_emb = cond_emb
		dec_cond_emb = cond_emb

		# index_mask = None
		# # index_mask = np.where(np.random.rand(cond_emb.size(0)) <= self.conf.mask_threshold)
		# index_mask = np.where(np.random.rand(cond_emb.size(0)) <= 1.)

		# hs = []
		hs = [[] for _ in range(len(self.conf.channel_mult))]

		if x is not None:
			h = x.type(self.dtype)

			# input blocks
			k = 0
			for i in range(len(self.input_num_blocks)):
				for j in range(self.input_num_blocks[i]):
					# modify zero arch
					# if k > 0:
					# 	tmp_zero = torch.zeros_like(self.input_blocks[k][0].out_layers[3].weight)
					# 	if torch.equal(tmp_zero, self.input_blocks[k][0].out_layers[3].weight):
					# 		torch.nn.init.normal_(self.input_blocks[k][0].out_layers[3].weight, 0, std)

					# tmp_zero_att = torch.zeros_like(self.input_blocks[4][1].proj_out.weight)
					# if torch.equal(tmp_zero_att, self.input_blocks[4][1].proj_out.weight):
					# 	torch.nn.init.normal_(self.input_blocks[4][1].proj_out.weight, 0, std)
					
					# tmp_zero_att = torch.zeros_like(self.input_blocks[5][1].proj_out.weight)
					# if torch.equal(tmp_zero_att, self.input_blocks[5][1].proj_out.weight):
					# 	torch.nn.init.normal_(self.input_blocks[5][1].proj_out.weight, 0, std)

					# print('enc_cond_emb size : ')
					# print(enc_cond_emb.size())
					
					h = self.input_blocks[k](h,
											emb=enc_time_emb,
											cond=enc_cond_emb)

					# print(i, j, h.shape)
					hs[i].append(h)
					k += 1
			assert k == len(self.input_blocks)

			# modify zero arch
			# tmp_zero = torch.zeros_like(self.middle_block[0].out_layers[3].weight)
			# if torch.equal(tmp_zero, self.middle_block[0].out_layers[3].weight):
			# 	torch.nn.init.normal_(self.middle_block[0].out_layers[3].weight, 0, std)
			# tmp_zero = torch.zeros_like(self.middle_block[2].out_layers[3].weight)
			# if torch.equal(tmp_zero, self.middle_block[2].out_layers[3].weight):
			# 	torch.nn.init.normal_(self.middle_block[2].out_layers[3].weight, 0, std)
			# tmp_zero = torch.zeros_like(self.middle_block[1].proj_out.weight)
			# if torch.equal(tmp_zero, self.middle_block[1].proj_out.weight):
			# 	torch.nn.init.normal_(self.middle_block[1].proj_out.weight, 0, std)

			# middle blocks
			h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
			# if cond.requires_grad:
			# 	print('h grad: ')
			# 	print(autograd.grad(h, cond, torch.ones_like(h)))
		else:
			# no lateral connections
			# happens when training only the autonecoder
			h = None
			hs = [[] for _ in range(len(self.conf.channel_mult))]

		# output blocks
		k = 0
		for i in range(len(self.output_num_blocks)):
			for j in range(self.output_num_blocks[i]):
				# take the lateral connection from the same layer (in reserve)
				# until there is no more, use None
				try:
					lateral = hs[-i - 1].pop()
					# print(i, j, lateral.shape)
				except IndexError:
					lateral = None
					# print(i, j, lateral)
				
				# modify zero arch
				# tmp_zero = torch.zeros_like(self.output_blocks[k][0].out_layers[3].weight)
				# if torch.equal(tmp_zero, self.output_blocks[k][0].out_layers[3].weight):
				# 	torch.nn.init.normal_(self.output_blocks[k][0].out_layers[3].weight, 0, std)

				h = self.output_blocks[k](h,
										emb=dec_time_emb,
										cond=dec_cond_emb,
										lateral=lateral,)
				k += 1

		pred = self.out(h)
		return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
	pred: Tensor
	cond: Tensor = None


class EmbedReturn(NamedTuple):
	# style and time
	emb: Tensor = None
	# time only
	time_emb: Tensor = None
	# style only (but could depend on time)
	style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
	# embed only style
	def __init__(self, time_channels, time_out_channels):
		super().__init__()
		self.time_embed = nn.Sequential(
			linear(time_channels, time_out_channels),
			nn.SiLU(),
			linear(time_out_channels, time_out_channels),
		)
		self.style = nn.Identity()

	def forward(self, time_emb=None, cond=None, **kwargs):
		if time_emb is None:
			# happens with autoenc training mode
			time_emb = None
		else:
			time_emb = self.time_embed(time_emb)
		style = self.style(cond)
		return EmbedReturn(emb=style, time_emb=time_emb, style=style)

class TimeStyleSeperateEmbed_decouple(nn.Module):
	# embed only style
	def __init__(self, time_channels, time_out_channels):
		super().__init__()
		self.time_embed = nn.Sequential(
			linear(time_channels, time_out_channels),
			nn.SiLU(),
			linear(time_out_channels, time_out_channels),
		)
		self.style = nn.Identity()

	def forward(self, time_emb=None, cond=None, index_mask=None, **kwargs):
		assert cond is not None
		z = cond[:, :256]
		s = cond[:, 256:]
		s_copy = torch.clone(s)
		# assert index_mask is not None
		if index_mask is not None and len(index_mask[0])>0:
			s_copy[index_mask] *= 0.
		if time_emb is None:
			# happens with autoenc training mode
			time_emb = None
		else:
			time_emb = self.time_embed(time_emb)
		time_emb += s_copy
		style = self.style(z)
		return EmbedReturn(emb=style, time_emb=time_emb, style=style)
