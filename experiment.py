import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *
import torch.optim as optim
from torchvision import transforms
import higher

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

debug_flag = 0


class DifferentiableAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DifferentiableAdam, self).__init__(params, defaults)
        self.fmodel = None
        self.diffopt = None

    def step(self, closure=None):
        """Performs a single optimization step (parameter update)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.fmodel is None or self.diffopt is None:
            raise ValueError("fmodel and diffopt should be set before calling step.")

        self.diffopt.step(loss)

        return loss

    def set_fmodel_diffopt(self, fmodel, diffopt):
        """Sets the functional model and differentiable optimizer."""
        self.fmodel = fmodel
        self.diffopt = diffopt


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def compute_acc(pred, target):
	return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]



class NetworkBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
		super(NetworkBlock, self).__init__()
		self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

	def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
		layers = []
		for i in range(int(nb_layers)):
			layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.layer(x)


class WideResNet(nn.Module):
	""" Based on code from https://github.com/yaodongyu/TRADES """

	def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
		super(WideResNet, self).__init__()

		num_input_channels = 3
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2471, 0.2435, 0.2616)
		self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
		self.std = torch.tensor(std).view(num_input_channels, 1, 1)

		nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
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

	def forward(self, x):
		# out = (x - self.mean.to(
		# x.device)) / self.std.to(x.device)
		out = x
		out = self.conv1(out)
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.relu(self.bn1(out))
		out = F.avg_pool2d(out, 8)
		out = out.view(-1, self.nChannels)
		return self.fc(out)


class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
							padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
							padding=1, bias=False)
		self.droprate = dropRate
		self.equalInOut = (in_planes == out_planes)
		self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
																padding=0, bias=False) or None

	def forward(self, x):
		if not self.equalInOut:
			x = self.relu1(self.bn1(x))
		else:
			out = self.relu1(self.bn1(x))
		out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, training=self.training)
		out = self.conv2(out)
		return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class LitModel(pl.LightningModule):
	def __init__(self, conf: TrainConfig):
		super().__init__()
		assert conf.train_mode != TrainMode.manipulate
		if conf.seed is not None:
			pl.seed_everything(conf.seed)

		self.save_hyperparameters(conf.as_dict_jsonable())

		self.conf = conf

		self.model = conf.make_model_conf().make_model()
		self.ema_model = copy.deepcopy(self.model)
		# self.ema_model.requires_grad_(False)
		self.ema_model.eval()

		model_size = 0
		for param in self.model.parameters():
			model_size += param.data.nelement()
		print('Model params: %.2f M' % (model_size / 1024 / 1024))

		self.sampler = conf.make_diffusion_conf().make_sampler()
		self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

		# this is shared for both model and latent
		self.T_sampler = conf.make_T_sampler()

		if conf.train_mode.use_latent_net():
			self.latent_sampler = conf.make_latent_diffusion_conf(
			).make_sampler()
			self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
			).make_sampler()
		else:
			self.latent_sampler = None
			self.eval_latent_sampler = None
		
		# self.uncond_sampler = conf.
		self.cls = WideResNet(depth=70, widen_factor=16, dropRate=0.3)
		state = torch.load('/home/users/zhangmingkun/diffae_causal/weights.pt')
		r = {}
		for k, v in list(state.items()):
			k = k.split('module.', 1)[1]
			r[k] = v
		self.cls.load_state_dict(r)

		# initial variables for consistent sampling
		self.register_buffer(
			'x_T',
			torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

		if conf.pretrain is not None:
			print(f'loading pretrain ... {conf.pretrain.name}')
			state = torch.load(conf.pretrain.path, map_location='cpu')
			print('step:', state['global_step'])
			self.load_state_dict(state['state_dict'], strict=False)

		if conf.latent_infer_path is not None:
			print('loading latent stats ...')
			state = torch.load(conf.latent_infer_path)
			self.conds = state['conds']
			self.register_buffer('conds_mean', state['conds_mean'][None, :])
			self.register_buffer('conds_std', state['conds_std'][None, :])
		else:
			self.conds_mean = None
			self.conds_std = None
		self.E = None

	def normalize(self, cond):
		cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
			self.device)
		return cond

	def denormalize(self, cond):
		cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
			self.device)
		return cond

	def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
		losses = 0
		unstable = 0
		dead = 0
		alive = 0
		
		# 处理 Enc_x 部分
		if norm == np.inf:
			mid = (x_U + x_L) / 2.0
			diff = (x_U - x_L) / 2.0
			enc_x_mid = self.ema_model.lacim.Enc_x(mid)
			enc_x_diff = self.ema_model.lacim.Enc_x(diff.abs()).abs()

			# 加上误差范围
			enc_x_U = enc_x_mid + enc_x_diff
			enc_x_L = enc_x_mid - enc_x_diff

		# 处理 encode_mu_var 和 reparameterization
		mu_U, logvar_U = self.ema_model.lacim.encode_mu_var(enc_x_U, env_idx=0)
		mu_L, logvar_L = self.ema_model.lacim.encode_mu_var(enc_x_L, env_idx=0)
		
		sigma_U = torch.exp(0.5 * logvar_U)
		sigma_L = torch.exp(0.5 * logvar_L)

		# 重参数化
		cond_U = mu_U + sigma_U * torch.randn_like(sigma_U)
		cond_L = mu_L + sigma_L * torch.randn_like(sigma_L)

		# 处理解码部分
		output_U = self.dec_y(cond_U[:, self.ema_model.lacim.s_dim:])
		output_L = self.dec_y(cond_L[:, self.ema_model.lacim.s_dim:])

		# 计算指标
		loss = ((output_U - output_L)**2).mean()
		losses += loss#.item()

		# 计算活性和死亡的神经元（简化示例）
		alive += torch.sum(output_U != output_L)
		dead += torch.sum(output_U == output_L)

		return output_U, output_L, losses, unstable, dead, alive

		
	def sample(self, N, device, T=None, T_latent=None):
		if T is None:
			sampler = self.eval_sampler
			latent_sampler = self.latent_sampler
		else:
			sampler = self.conf._make_diffusion_conf(T).make_sampler()
			latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

		noise = torch.randn(N,
							3,
							self.conf.img_size,
							self.conf.img_size,
							device=device)
		pred_img = render_uncondition(
			self.conf,
			self.ema_model,
			noise,
			sampler=sampler,
			latent_sampler=latent_sampler,
			conds_mean=self.conds_mean,
			conds_std=self.conds_std,
		)
		pred_img = (pred_img + 1) / 2
		return pred_img

	def render(self, noise, cond=None, T=None):
		if T is None:
			sampler = self.eval_sampler
		else:
			sampler = self.conf._make_diffusion_conf(T).make_sampler()

		if cond is not None:
			pred_img = render_condition(self.conf,
										self.ema_model,
										noise,
										sampler=sampler,
										cond=cond)
		else:
			pred_img = render_uncondition(self.conf,
										self.ema_model,
										noise,
										sampler=sampler,
										latent_sampler=None)
		pred_img = (pred_img + 1) / 2
		return pred_img

	def encode(self, x, r=0):
		# TODO:
		# assert self.conf.model_type.has_autoenc()
		# cond = self.ema_model.encoder.forward(x)
		# return cond
		x_enc = self.ema_model.lacim.Enc_x(x)
		mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
		mu_prior, logvar_prior = self.ema_model.lacim.encode_prior(x_enc, env_idx=0)
		# if r == 1:
		# 	logvar = 2. * logvar
		# elif r == 2:
		# 	logvar = logvar / 4.
		cond = self.ema_model.reparameterize(mu, logvar)

		# x_enc = self.model.lacim.Enc_x(x)
		# mu, logvar = self.model.lacim.encode_mu_var(x_enc, env_idx=0)
		# mu_prior, logvar_prior = self.model.lacim.encode_prior(x_enc, env_idx=0)
		# cond = self.model.reparameterize(mu, logvar)

		# return cond
		return {'cond':cond}
	
	
	def search_rv(self, x):
		with torch.no_grad():
			x_enc = self.ema_model.lacim.Enc_x(x)
			mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
			mu_prior, logvar_prior = self.ema_model.lacim.encode_prior(x_enc, env_idx=0)
			cond_init = self.ema_model.reparameterize(mu, logvar)

		# x_enc = self.model.lacim.Enc_x(x)
		# mu, logvar = self.model.lacim.encode_mu_var(x_enc, env_idx=0)
		# mu_prior, logvar_prior = self.model.lacim.encode_prior(x_enc, env_idx=0)
		# cond_init = self.model.reparameterize(mu, logvar)
		
		cond = cond_init
		# cond = torch.randn(x.size(0), 512).to(x.device)
		cond.requires_grad = True
		# # cond = torch.randn_like(cond_init)
		# cond.requires_grad = True
		optimizer = optim.Adam(params=[cond], lr=self.conf.lr_search, weight_decay=self.conf.reg_search)
		
		
		for i in range(self.conf.ep_search):
			optimizer.zero_grad()
			nll = 0.
			nll_uncond = 0.

			for tt in range(self.conf.t_N_search):
				t = torch.tensor([tt*90 for _ in range(x.size(0))])
				t = t.long()
				t = t.to(x.device)
				with torch.enable_grad():
					# t, weight = self.T_sampler.sample(len(x), x.device)
					# t = torch.tensor([tt for _ in range(x.size(0))])
					# t = self.uniform_noise((x.size(0),), 400, 600)
					# t = t.long()
					t = t.to(x.device)
					losses = self.sampler.training_losses(model=self.ema_model,
														x_start=x,
														t=t,
														cond=cond,
														mask_threshold=0.)

					# losses_uncond = self.sampler.training_losses(model=self.ema_model,
					# 									x_start=x,
					# 									t=t,
					# 									cond=cond,
					# 									mask_threshold=1.)
					# likelihood = losses['likelihood']

					# if likelihood.requires_grad is False:
						# likelihood.requires_grad = True

					# likelihood_all += likelihood
					nll += losses['likelihood'].mean()
					# nll_uncond += losses_uncond['likelihood'].mean()
			loss = nll / self.conf.t_N_size
			if loss.requires_grad is False:
				loss.requires_grad = True
			
			loss.backward(retain_graph=True)
			
			optimizer.step()
		return cond


	def search_rv_with_grad(self, x):
		with torch.enable_grad():
			x_enc = self.ema_model.lacim.Enc_x(x)
			mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
			mu_prior, logvar_prior = self.ema_model.lacim.encode_prior(x_enc, env_idx=0)
			cond = self.ema_model.reparameterize(mu, logvar)
			# return cond

			# cond = torch.randn_like(cond_init)
			# cond = torch.randn(x.size(0), 512).to(x.device)
			# cond.requires_grad = True
			# optimizer = optim.Adam(params=[cond], lr=self.conf.lr_search, weight_decay=self.conf.reg_search)
			
			# t, _ = self.T_sampler.sample(len(x), x.device)
			for i in range(self.conf.ep_search):
			# for i in range(0):
				# optimizer.zero_grad()
				nll = 0.
				# likelihood_all = None
				if debug_flag == 1:
					pred = self.dec_y(cond[:, self.ema_model.lacim.s_dim:])
					pred = np.array(pred.detach().cpu().numpy()).reshape((x.size(0), self.conf.num_classes))
					pred_y = np.argmax(pred, axis=1)
					print('pred_y : {}'.format(pred_y))
				# t_all = torch.arange(self.conf.t_N_search)
				# t_all = t_all.split(self.conf.t_N_size, dim=0)
				for tt in range(self.conf.t_N_search):
				# for tt in t_all:
					# tt = tt.long()
					# t = tt.to(x.device)
					# t = torch.tensor([tt*90 for _ in range(x.size(0))])
					t = torch.tensor([tt*20 for _ in range(x.size(0))])
					t = t.long()
					t = t.to(x.device)
					with torch.enable_grad():
						# # t, _ = self.T_sampler.sample(len(x), x.device)
						# t = self.uniform_noise((x.size(0),), 400, 600)
						# # print(x.requires_grad)
						losses = self.sampler.training_losses(model=self.ema_model,
															x_start=x,
															t=t,
															cond=cond,
															mask_threshold=0.)
						
						losses_uncond = self.sampler.training_losses(model=self.ema_model,
															x_start=x,
															t=t,
															cond=cond,
															mask_threshold=1.)
						likelihood = losses['likelihood']
						likelihood_uncond = losses_uncond['likelihood']
						# if likelihood_all is None:
						# 	likelihood_all = likelihood
						# else:
						# 	likelihood_all += likelihood
						if debug_flag == 1:
							print('likelihood of search_rv_adv = {} '.format(likelihood.tolist()))
							print('likelihood_uncond = {} '.format(likelihood_uncond.tolist()))
							print()
						# if likelihood.requires_grad is False:
						# 	likelihood.requires_grad = True
						nll += likelihood.mean()
				loss = nll / self.conf.t_N_search
				grad_cond = torch.autograd.grad(loss, cond, torch.ones_like(loss), retain_graph=True)[0].clone()
				cond = cond - self.conf.lr_search * grad_cond.sign()
			return cond
	

	def get_E(self):
		return self.E

	def uniform_noise(self, size, begin=0.0, end=1.0):
		x = torch.rand(size)
		x = x * (end - begin) + begin
		return x



	def search_ben(self,x):
		with torch.enable_grad():
			E = torch.zeros_like(x)
			E = E.to(x.device)
			E.requires_grad = True
			x_ben = x - E
			momentum = torch.zeros_like(x)
			x_enc = self.ema_model.lacim.Enc_x(x_ben)
			mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
			mu_prior, logvar_prior = self.ema_model.lacim.encode_prior(x_enc, env_idx=0)
			cond = self.ema_model.reparameterize(mu, logvar)
			# optimizer = DifferentiableAdam([E], lr=0.05)
			
			# t, _ = self.T_sampler.sample(len(x_ben), x.device)
			# t = self.uniform_noise((x.size(0),), 0, 100)
			
			# lr_decay
			# t = torch.tensor([0 for _ in range(x_ben.size(0))])
			# # t = self.uniform_noise((x.size(0),), 0, 50)
			# t = t.long()
			# t = t.to(x.device)
			
			# decay = torch.ones_like(t)
			# # decay_low = decay * 1e-3
			# # losses_uncond = self.sampler.training_losses(model=self.ema_model, x_start=x_ben, t=t, cond=cond, mask_threshold=1.)['likelihood']
			# # lr_decay = torch.where(losses_uncond < 0.8, torch.tensor(decay_low), decay)
			# lr_decay = decay

			# # 扩展为与x维度一致，即[bs] -> [bs, 3, 32, 32]
			# lr_decay = lr_decay.unsqueeze(1).unsqueeze(2).unsqueeze(3)
			# lr_decay = lr_decay.expand(-1, 3, 32, 32)
			for i in range(self.conf.ep_search_E):
				# optimizer.zero_grad()
				nll = 0.
				# for tt in range(self.conf.t_N_search):
				# t, _ = self.T_sampler.sample(len(x_ben), x.device)
				for tt in range(1):
					with torch.enable_grad():
						# t, _ = self.T_sampler.sample(len(x_ben), x.device)
						# t = torch.tensor([tt for _ in range(x.size(0))])
						t = self.uniform_noise((x.size(0),), 0, 50)
						t = t.long()
						t = t.to(x.device)

						losses_uncond = self.sampler.training_losses(model=self.ema_model,
															x_start=x_ben,
															t=t,
															cond=cond,
															mask_threshold=1.)

						# losses_per = self.sampler.training_losses(model=self.ema_model,
						# 									x_start=x_ben,
						# 									t=t,
						# 									cond=torch.zeros_like(cond))
						# likelihood = (losses['likelihood'] + losses_per['likelihood']) / 2.
						# if likelihood_all is None:
						# 	likelihood_all = likelihood
						# else:
						# 	likelihood_all += likelihood
						# print('likelihood of search_rv_adv = {} '.format(likelihood))
						# if likelihood.requires_grad is False:
						# 	likelihood.requires_grad = True
						nll += losses_uncond['likelihood'].mean()
						# nll += losses_uncond['likelihood'].mean()

				# loss = nll / self.conf.t_N_search
				loss = nll
				# print('i = {}, \t loss = {:.2f}, \n \t loss_uncond = {:.2f}'.format(i, losses['likelihood'].mean(), losses_uncond['likelihood'].mean()))
				if debug_flag == 1:
					print('i = {}'.format(i))
					# # print('\t loss = ', end="")
					# # print(losses['likelihood'].tolist())
					print('\t loss_uncond = ', end="")
					print(losses_uncond['likelihood'].tolist())

				grad_E = torch.autograd.grad(loss, E, torch.ones_like(loss), retain_graph=True)[0].clone()
				# E.grad = grad_E
				# E = optimizer.step()[0]["params"][0]
				momentum = momentum - grad_E / torch.norm(grad_E, p=1)

				# E = E - self.conf.lr_search_E * 1. * grad_E.sign()
				# print(lr_decay.size())
				# print(momentum.size())
				# print(E.size())
				E = E + self.conf.lr_search_E * 1. * momentum.sign() # * lr_decay
				# E = E + self.conf.lr_search_E * 1. * momentum
				# optimizer.step()
				# E = torch.clamp(E, -8/255., 8/255.)
				x_ben = x - E

				if debug_flag == 1:
					x_enc = self.ema_model.lacim.Enc_x(x_ben)
					mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
					cond = self.ema_model.reparameterize(mu, logvar)
					pred = self.dec_y(cond[:, self.ema_model.lacim.s_dim:])
					pred = np.array(pred.detach().cpu().numpy()).reshape((x.size(0), self.conf.num_classes))
					pred_y = np.argmax(pred, axis=1)
					print('pred_y : {}'.format(pred_y))
			# clamp 
			# E = torch.clamp(E, -8/255., 8/255.)
			# print(E)
			# assert i == 1
			# x_ben = x - E
			x_ben = torch.clamp(x_ben, -1., 1.)

			if debug_flag == 1:
				print('-' * 70)
			self.E = E.detach()
			return x_ben#.detach()


	def search_ben_wograd(self,x):
		with torch.enable_grad():
			E = torch.zeros_like(x)
			E = E.to(x.device)
			E.requires_grad = True
			x_ben = x - E
			momentum = torch.zeros_like(x)
			x_enc = self.ema_model.lacim.Enc_x(x_ben)
			mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
			mu_prior, logvar_prior = self.ema_model.lacim.encode_prior(x_enc, env_idx=0)
			cond = self.ema_model.reparameterize(mu, logvar)

			for i in range(self.conf.ep_search_E):
				# optimizer.zero_grad()
				nll = 0.
				# for tt in range(self.conf.t_N_search):
				# t, _ = self.T_sampler.sample(len(x_ben), x.device)
				for tt in range(1):
					with torch.enable_grad():
						# t, _ = self.T_sampler.sample(len(x_ben), x.device)
						# t = torch.tensor([tt for _ in range(x.size(0))])
						t = self.uniform_noise((x.size(0),), 0, 50)
						t = t.long()
						t = t.to(x.device)

						losses_uncond = self.sampler.training_losses(model=self.ema_model,
															x_start=x_ben,
															t=t,
															cond=cond,
															mask_threshold=1.)

						# losses_per = self.sampler.training_losses(model=self.ema_model,
						# 									x_start=x_ben,
						# 									t=t,
						# 									cond=torch.zeros_like(cond))
						# likelihood = (losses['likelihood'] + losses_per['likelihood']) / 2.
						# if likelihood_all is None:
						# 	likelihood_all = likelihood
						# else:
						# 	likelihood_all += likelihood
						# print('likelihood of search_rv_adv = {} '.format(likelihood))
						# if likelihood.requires_grad is False:
						# 	likelihood.requires_grad = True
						nll += losses_uncond['likelihood'].mean()
						# nll += losses_uncond['likelihood'].mean()

				# loss = nll / self.conf.t_N_search
				loss = nll
				# print('i = {}, \t loss = {:.2f}, \n \t loss_uncond = {:.2f}'.format(i, losses['likelihood'].mean(), losses_uncond['likelihood'].mean()))
				if debug_flag == 1:
					print('i = {}'.format(i))
					# # print('\t loss = ', end="")
					# # print(losses['likelihood'].tolist())
					print('\t loss_uncond = ', end="")
					print(losses_uncond['likelihood'].tolist())

				grad_E = torch.autograd.grad(loss, E, torch.ones_like(loss), allow_unused=True)[0].clone()
				# E.grad = grad_E
				# E = optimizer.step()[0]["params"][0]
				momentum = momentum - grad_E / torch.norm(grad_E, p=1)

				# E = E - self.conf.lr_search_E * 1. * grad_E.sign()
				# print(lr_decay.size())
				# print(momentum.size())
				# print(E.size())
				E = E + self.conf.lr_search_E * 1. * momentum.sign() #* lr_decay
				# E = E + self.conf.lr_search_E * 1. * momentum
				# optimizer.step()
				E = torch.clamp(E, -8/255. * 2, 8/255. * 2)
				x_ben = x - E

				if debug_flag == 1:
					x_enc = self.ema_model.lacim.Enc_x(x_ben)
					mu, logvar = self.ema_model.lacim.encode_mu_var(x_enc, env_idx=0)
					cond = self.ema_model.reparameterize(mu, logvar)
					pred = self.dec_y(cond[:, self.ema_model.lacim.s_dim:])
					pred = np.array(pred.detach().cpu().numpy()).reshape((x.size(0), self.conf.num_classes))
					pred_y = np.argmax(pred, axis=1)
					print('pred_y : {}'.format(pred_y))
			# clamp 
			# E = torch.clamp(E, -8/255., 8/255.)
			# print(E)
			# assert i == 1
			x_ben = x - E
			x_ben = torch.clamp(x_ben, -1., 1.)

			if debug_flag == 1:
				print('-' * 70)
			self.E = E.detach()
			return x_ben#.detach()


	def dec_y(self, s):
		# return self.model.lacim.Dec_y(s)
		return self.ema_model.lacim.Dec_y(s)

	def encode_stochastic(self, x, cond, T=None):
		if T is None:
			sampler = self.eval_sampler
		else:
			sampler = self.conf._make_diffusion_conf(T).make_sampler()
		out = sampler.ddim_reverse_sample_loop(self.ema_model,
											x,
											model_kwargs={'cond': cond})
		return out['sample']

	def inference(self, x):
		# x = (x - 0.5) * 2
		cond = self.search_rv(x)
		# return self.dec_y(cond)
		return self.dec_y(cond[:, self.ema_model.lacim.s_dim:])

	def inference_cls(self, x, req_grad=True):
		x = (x - 0.5) * 2
		# x = normalize(x)
		# if req_grad:
		if x.requires_grad and req_grad:
		# 	# cond = self.search_rv_adv(x)
			x_ben = self.search_ben(x)
		# 	# cond = self.search_rv_with_grad(x_ben)
			return self.cls(x_ben)
		else:
			x_ben = self.search_ben_wograd(x)
		# 	# cond = self.search_rv(x)
			return self.cls(x_ben)

	
	def inference_cls_only(self, x, req_grad=True):
		x = (x - 0.5) * 2
		return self.cls(x)

	def inference_causal(self, x, req_grad=True):
		x = (x - 0.5) * 2
		if x.requires_grad and req_grad:
			cond = self.search_rv_with_grad(x)
		else:
			cond = self.search_rv(x)
		return self.dec_y(cond[:, self.ema_model.lacim.s_dim:])

	def inference_causal_purify(self, x, req_grad=True):
		x = (x - 0.5) * 2
		if x.requires_grad and req_grad:
			x_ben = self.search_ben(x)
			cond = self.search_rv_with_grad(x_ben)
			# print(self.dec_y(cond[:, self.ema_model.lacim.s_dim:]))
		else:
			x_ben = self.search_ben_wograd(x)
			cond = self.search_rv(x_ben)
		return self.dec_y(cond[:, self.ema_model.lacim.s_dim:])
	
	def forward(self, x, req_grad=True):
		x = (x - 0.5) * 2
		if x.requires_grad and req_grad:
			x_ben = self.search_ben(x)
			cond = self.search_rv_with_grad(x_ben)
			# print(self.dec_y(cond[:, self.ema_model.lacim.s_dim:]))
		else:
			x_ben = self.search_ben_wograd(x)
			cond = self.search_rv(x_ben)
		return self.dec_y(cond[:, self.ema_model.lacim.s_dim:])


	def inference_adv(self, x):
		# x = (x - 0.5) * 2
		cond = self.search_rv_adv(x)
		# return self.dec_y(cond)
		return self.dec_y(cond[:, self.ema_model.lacim.s_dim:])

	def setup(self, stage=None) -> None:
		"""
		make datasets & seeding each worker separately
		"""
		##############################################
		# NEED TO SET THE SEED SEPARATELY HERE
		if self.conf.seed is not None:
			seed = self.conf.seed * get_world_size() + self.global_rank
			np.random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			print('local seed:', seed)
		##############################################

		self.train_data = self.conf.make_dataset()
		print('train data:', len(self.train_data))
		self.val_data = self.train_data
		print('val data:', len(self.val_data))

	def _train_dataloader(self, drop_last=True):
		"""
		really make the dataloader
		"""
		# make sure to use the fraction of batch size
		# the batch size is global!
		conf = self.conf.clone()
		conf.batch_size = self.batch_size

		dataloader = conf.make_loader(self.train_data,
									shuffle=True,
									drop_last=drop_last)
		return dataloader

	def train_dataloader(self):
		"""
		return the dataloader, if diffusion mode => return image dataset
		if latent mode => return the inferred latent dataset
		"""
		print('on train dataloader start ...')
		if self.conf.train_mode.require_dataset_infer():
			if self.conds is None:
				# usually we load self.conds from a file
				# so we do not need to do this again!
				self.conds = self.infer_whole_dataset()
				# need to use float32! unless the mean & std will be off!
				# (1, c)
				self.conds_mean.data = self.conds.float().mean(dim=0,
															keepdim=True)
				self.conds_std.data = self.conds.float().std(dim=0,
															keepdim=True)
			print('mean:', self.conds_mean.mean(), 'std:',
				self.conds_std.mean())

			# return the dataset with pre-calculated conds
			conf = self.conf.clone()
			conf.batch_size = self.batch_size
			data = TensorDataset(self.conds)
			return conf.make_loader(data, shuffle=True)
		else:
			return self._train_dataloader()

	@property
	def batch_size(self):
		"""
		local batch size for each worker
		"""
		ws = get_world_size()
		assert self.conf.batch_size % ws == 0
		return self.conf.batch_size // ws

	@property
	def num_samples(self):
		"""
		(global) batch size * iterations
		"""
		# batch size here is global!
		# global_step already takes into account the accum batches
		return self.global_step * self.conf.batch_size_effective

	def is_last_accum(self, batch_idx):
		"""
		is it the last gradient accumulation loop? 
		used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
		"""
		return (batch_idx + 1) % self.conf.accum_batches == 0

	def infer_whole_dataset(self,
							with_render=False,
							T_render=None,
							render_save_path=None):
		"""
		predicting the latents given images using the encoder

		Args:
			both_flips: include both original and flipped images; no need, it's not an improvement
			with_render: whether to also render the images corresponding to that latent
			render_save_path: lmdb output for the rendered images
		"""
		data = self.conf.make_dataset()
		if isinstance(data, CelebAlmdb) and data.crop_d2c:
			# special case where we need the d2c crop
			data.transform = make_transform(self.conf.img_size,
											flip_prob=0,
											crop_d2c=True)
		else:
			data.transform = make_transform(self.conf.img_size, flip_prob=0)

		# data = SubsetDataset(data, 21)

		loader = self.conf.make_loader(
			data,
			shuffle=False,
			drop_last=False,
			batch_size=self.conf.batch_size_eval,
			parallel=True,
		)
		model = self.ema_model
		model.eval()
		conds = []

		if with_render:
			sampler = self.conf._make_diffusion_conf(
				T=T_render or self.conf.T_eval).make_sampler()

			if self.global_rank == 0:
				writer = LMDBImageWriter(render_save_path,
										format='webp',
										quality=100)
			else:
				writer = nullcontext()
		else:
			writer = nullcontext()

		with writer:
			for batch in tqdm(loader, total=len(loader), desc='infer'):
				# with torch.no_grad():
				# (n, c)
				# print('idx:', batch['index'])
				# cond = model.encoder(batch['img'].to(self.device))
				cond = self.encode(batch['img'].to(self.device))['cond']

				# used for reordering to match the original dataset
				idx = batch['index']
				idx = self.all_gather(idx)
				if idx.dim() == 2:
					idx = idx.flatten(0, 1)
				argsort = idx.argsort()

				if with_render:
					noise = torch.randn(len(cond),
										3,
										self.conf.img_size,
										self.conf.img_size,
										device=self.device)
					render = sampler.sample(model, noise=noise, cond=cond)
					render = (render + 1) / 2
					# print('render:', render.shape)
					# (k, n, c, h, w)
					render = self.all_gather(render)
					if render.dim() == 5:
						# (k*n, c)
						render = render.flatten(0, 1)

					# print('global_rank:', self.global_rank)

					if self.global_rank == 0:
						writer.put_images(render[argsort])

				# (k, n, c)
				cond = self.all_gather(cond)

				if cond.dim() == 3:
					# (k*n, c)
					cond = cond.flatten(0, 1)

				conds.append(cond[argsort].cpu())
			# break
		model.train()
		# (N, c) cpu

		conds = torch.cat(conds).float()
		return conds

	def training_step(self, batch, batch_idx):
		"""
		given an input, calculate the loss function
		no optimization at this stage.
		"""
		with amp.autocast(False):
			# batch size here is local!
			# forward
			if self.conf.train_mode.require_dataset_infer():
				# this mode as pre-calculated cond
				cond = batch[0]
				# cond = torch.cat([torch.zeros_like(cond[:, :256]), cond[:, 256:]], dim=1)
				if self.conf.latent_znormalize:
					# if len(self.conds_mean.size()) > 1:
						# self.conds_mean = self.conds_mean.mean()
						# self.conds_std = self.conds_std.mean()
					# cond = cond[:, 256:]
					cond = (cond - self.conds_mean.to(
						self.device)) / self.conds_std.to(self.device)
			else:
				imgs, idxs = batch['img'], batch['index']
				# r = batch['type']
				# print(f'(rank {self.global_rank}) batch size:', len(imgs))
				x_start = imgs
				target = batch['target']			

			if self.conf.train_mode == TrainMode.diffusion:
				"""
				main training mode!!!
				"""
				# with numpy seed we have the problem that the sample t's are related!
				t, weight = self.T_sampler.sample(len(x_start), x_start.device)
				losses = self.sampler.training_losses(model=self.model,
													x_start=x_start,
													t=t,
													target_y=target,)
			elif self.conf.train_mode.is_latent_diffusion():
				"""
				training the latent variables!
				"""
				# diffusion on the latent
				
				t, weight = self.T_sampler.sample(len(cond), cond.device)
				latent_losses = self.latent_sampler.training_losses(
					model=self.model.latent_net, x_start=cond, t=t, is_latent=1)
				# train only do the latent diffusion
				losses = {
					'latent': latent_losses['loss'],
					'loss': latent_losses['loss']
				}
			else:
				raise NotImplementedError()

			loss = losses['loss'].mean()
			# divide by accum batches to make the accumulated gradient exact!
			for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
				if key in losses:
					losses[key] = self.all_gather(losses[key]).mean()

			if self.global_rank == 0:
				self.logger.experiment.add_scalar('loss', losses['loss'],
												self.num_samples)
				for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
					if key in losses:
						self.logger.experiment.add_scalar(
							f'loss/{key}', losses[key], self.num_samples)

		return {'loss': loss}

	def on_train_batch_end(self, outputs, batch, batch_idx: int,
						dataloader_idx: int) -> None:
		"""
		after each training step ...
		"""
		if self.is_last_accum(batch_idx):
			# only apply ema on the last gradient accumulation step,
			# if it is the iteration that has optimizer.step()
			if self.conf.train_mode == TrainMode.latent_diffusion:
				# it trains only the latent hence change only the latent
				ema(self.model.latent_net, self.ema_model.latent_net,
					self.conf.ema_decay)
			else:
				ema(self.model, self.ema_model, self.conf.ema_decay)

			# logging
			if self.conf.train_mode.require_dataset_infer():
				imgs = None
			else:
				imgs = batch['img']
			self.log_sample(x_start=imgs)
			# self.evaluate_scores()

	def on_before_optimizer_step(self, optimizer: Optimizer,
								optimizer_idx: int) -> None:
		# fix the fp16 + clip grad norm problem with pytorch lightinng
		# this is the currently correct way to do it
		if self.conf.grad_clip > 0:
			# from trainer.params_grads import grads_norm, iter_opt_params
			params = [
				p for group in optimizer.param_groups for p in group['params']
			]
			# print('before:', grads_norm(iter_opt_params(optimizer)))
			torch.nn.utils.clip_grad_norm_(params,
										max_norm=self.conf.grad_clip)
			# print('after:', grads_norm(iter_opt_params(optimizer)))

	def log_sample(self, x_start):
		

		def do(model,
			postfix,
			use_xstart,
			save_real=False,
			no_latent_diff=False,
			interpolate=False):
			
			if self.global_step % 5e4 == 0:
				ckpt_dir = os.path.join(self.conf.logdir, f'ckpt')
				if not os.path.exists(ckpt_dir):
						os.makedirs(ckpt_dir)
				path = os.path.join(ckpt_dir, '%d.ckpt' % self.global_step)
				if not os.path.exists(path):
					torch.save(self.state_dict(), path)
					print('step = {} checkpoint saved!'.format(self.global_step))
				else:
					path = os.path.join(ckpt_dir, '%d_ema.ckpt' % self.global_step)
					torch.save(self.state_dict(), path)
					print('path already exists!')

			self.model.train()
			self.ema_model.train()
			if self.global_step % 5e4 != 0:
				return

			model.eval()

			# with torch.no_grad():
			all_x_T = self.split_tensor(self.x_T)
			batch_size = min(len(all_x_T), self.conf.batch_size_eval)
			# allow for superlarge models
			loader = DataLoader(all_x_T, batch_size=batch_size)

			Gen = []
			for x_T in loader:
				if use_xstart:
					_xstart = x_start[:len(x_T)]
				else:
					_xstart = None

				if self.conf.train_mode.is_latent_diffusion(
				) and not use_xstart:
					# diffusion of the latent first
					gen = render_uncondition(
						conf=self.conf,
						model=model,
						x_T=x_T,
						sampler=self.eval_sampler,
						latent_sampler=self.eval_latent_sampler,
						conds_mean=self.conds_mean,
						conds_std=self.conds_std)
				else:
					if not use_xstart and self.conf.model_type.has_noise_to_cond(
					):
						model: BeatGANsAutoencModel
						# special case, it may not be stochastic, yet can sample
						cond = torch.randn(len(x_T),
										self.conf.style_ch,
										device=self.device)
						cond = model.noise_to_cond(cond)
					else:
						if interpolate:
							with amp.autocast(self.conf.fp16):
								cond = model.encoder(_xstart)
								i = torch.randperm(len(cond))
								cond = (cond + cond[i]) / 2
						else:
							cond = None
					gen = self.eval_sampler.sample(model=model,
												noise=x_T,
												cond=cond,
												x_start=_xstart)
				Gen.append(gen)

			gen = torch.cat(Gen)
			gen = self.all_gather(gen)
			if gen.dim() == 5:
				# (n, c, h, w)
				gen = gen.flatten(0, 1)

			if save_real and use_xstart:
				# save the original images to the tensorboard
				real = self.all_gather(_xstart)
				if real.dim() == 5:
					real = real.flatten(0, 1)

				if self.global_rank == 0:
					grid_real = (make_grid(real) + 1) / 2
					self.logger.experiment.add_image(
						f'sample{postfix}/real', grid_real,
						self.num_samples)

			if self.global_rank == 0:
				# save samples to the tensorboard
				grid = (make_grid(gen) + 1) / 2
				sample_dir = os.path.join(self.conf.logdir,
										f'sample{postfix}')
				if not os.path.exists(sample_dir):
					os.makedirs(sample_dir)
				path = os.path.join(sample_dir,
									'%d.png' % self.num_samples)
				save_image(grid, path)
				self.logger.experiment.add_image(f'sample{postfix}', grid,
												self.num_samples)
			model.train()

		if self.conf.sample_every_samples > 0 and is_time(
				self.num_samples, self.conf.sample_every_samples,
				self.conf.batch_size_effective):

			if self.conf.train_mode.require_dataset_infer():
				do(self.model, '', use_xstart=False)
				do(self.ema_model, '_ema', use_xstart=False)
			else:
				if self.conf.model_type.has_autoenc(
				) and self.conf.model_type.can_sample():
					do(self.model, '', use_xstart=False)
					do(self.ema_model, '_ema', use_xstart=False)
					# autoencoding mode
					do(self.model, '_enc', use_xstart=True, save_real=True)
					do(self.ema_model,
					'_enc_ema',
					use_xstart=True,
					save_real=True)
				elif self.conf.train_mode.use_latent_net():
					do(self.model, '', use_xstart=False)
					do(self.ema_model, '_ema', use_xstart=False)
					# autoencoding mode
					do(self.model, '_enc', use_xstart=True, save_real=True)
					do(self.model,
					'_enc_nodiff',
					use_xstart=True,
					save_real=True,
					no_latent_diff=True)
					do(self.ema_model,
					'_enc_ema',
					use_xstart=True,
					save_real=True)
				else:
					do(self.model, '', use_xstart=True, save_real=True)
					do(self.ema_model, '_ema', use_xstart=True, save_real=True)

	def evaluate_scores(self):
		"""
		evaluate FID and other scores during training (put to the tensorboard)
		For, FID. It is a fast version with 5k images (gold standard is 50k).
		Don't use its results in the paper!
		"""
		def fid(model, postfix):
			score = evaluate_fid(self.eval_sampler,
								model,
								self.conf,
								device=self.device,
								train_data=self.train_data,
								val_data=self.val_data,
								latent_sampler=self.eval_latent_sampler,
								conds_mean=self.conds_mean,
								conds_std=self.conds_std)
			if self.global_rank == 0:
				self.logger.experiment.add_scalar(f'FID{postfix}', score,
												self.num_samples)
				if not os.path.exists(self.conf.logdir):
					os.makedirs(self.conf.logdir)
				with open(os.path.join(self.conf.logdir, 'eval.txt'),
						'a') as f:
					metrics = {
						f'FID{postfix}': score,
						'num_samples': self.num_samples,
					}
					f.write(json.dumps(metrics) + "\n")

		def lpips(model, postfix):
			if self.conf.model_type.has_autoenc(
			) and self.conf.train_mode.is_autoenc():
				# {'lpips', 'ssim', 'mse'}
				score = evaluate_lpips(self.eval_sampler,
									model,
									self.conf,
									device=self.device,
									val_data=self.val_data,
									latent_sampler=self.eval_latent_sampler)

				if self.global_rank == 0:
					for key, val in score.items():
						self.logger.experiment.add_scalar(
							f'{key}{postfix}', val, self.num_samples)

		if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
				self.num_samples, self.conf.eval_every_samples,
				self.conf.batch_size_effective):
			print(f'eval fid @ {self.num_samples}')
			lpips(self.model, '')
			fid(self.model, '')

		if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
				self.num_samples, self.conf.eval_ema_every_samples,
				self.conf.batch_size_effective):
			print(f'eval fid ema @ {self.num_samples}')
			fid(self.ema_model, '_ema')
			# it's too slow
			# lpips(self.ema_model, '_ema')

	def configure_optimizers(self):
		out = {}
		if self.conf.optimizer == OptimizerType.adam:
			# for name, param in self.model.named_parameters():
			# 	print('name = {}'.format(name))
			params_to_optimize = [param for name, param in self.model.named_parameters() if 'club' not in name]
			optim = torch.optim.Adam(params_to_optimize,
									lr=self.conf.lr,
									weight_decay=self.conf.weight_decay)
		elif self.conf.optimizer == OptimizerType.adamw:
			optim = torch.optim.AdamW(self.model.parameters(),
									lr=self.conf.lr,
									weight_decay=self.conf.weight_decay)
		else:
			raise NotImplementedError()
		out['optimizer'] = optim
		if self.conf.warmup > 0:
			sched = torch.optim.lr_scheduler.LambdaLR(optim,
													lr_lambda=WarmupLR(
														self.conf.warmup))
			out['lr_scheduler'] = {
				'scheduler': sched,
				'interval': 'step',
			}
		return out

	def split_tensor(self, x):
		"""
		extract the tensor for a corresponding "worker" in the batch dimension

		Args:
			x: (n, c)

		Returns: x: (n_local, c)
		"""
		n = len(x)
		rank = self.global_rank
		world_size = get_world_size()
		# print(f'rank: {rank}/{world_size}')
		per_rank = n // world_size
		return x[rank * per_rank:(rank + 1) * per_rank]

	def test_step(self, batch, *args, **kwargs):
		"""
		for the "eval" mode. 
		We first select what to do according to the "conf.eval_programs". 
		test_step will only run for "one iteration" (it's a hack!).
		
		We just want the multi-gpu support. 
		"""
		# make sure you seed each worker differently!
		self.setup()

		# it will run only one step!
		print('global step:', self.global_step)
		"""
		"infer" = predict the latent variables using the encoder on the whole dataset
		"""
		if 'infer' in self.conf.eval_programs:
			if 'infer' in self.conf.eval_programs:
				print('infer ...')
				conds = self.infer_whole_dataset().float()
				conds = conds[:,256:]
				# NOTE: always use this path for the latent.pkl files
				save_path = f'checkpoints/{self.conf.name}/latent.pkl'
			else:
				raise NotImplementedError()

			if self.global_rank == 0:
				conds_mean = conds.mean(dim=0)
				conds_std = conds.std(dim=0)
				if not os.path.exists(os.path.dirname(save_path)):
					os.makedirs(os.path.dirname(save_path))
				torch.save(
					{
						'conds': conds,
						'conds_mean': conds_mean,
						'conds_std': conds_std,
					}, save_path)
		"""
		"infer+render" = predict the latent variables using the encoder on the whole dataset
		THIS ALSO GENERATE CORRESPONDING IMAGES
		"""
		# infer + reconstruction quality of the input
		for each in self.conf.eval_programs:
			if each.startswith('infer+render'):
				m = re.match(r'infer\+render([0-9]+)', each)
				if m is not None:
					T = int(m[1])
					self.setup()
					print(f'infer + reconstruction T{T} ...')
					conds = self.infer_whole_dataset(
						with_render=True,
						T_render=T,
						render_save_path=
						f'latent_infer_render{T}/{self.conf.name}.lmdb',
					)
					save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
					conds_mean = conds.mean(dim=0)
					conds_std = conds.std(dim=0)
					if not os.path.exists(os.path.dirname(save_path)):
						os.makedirs(os.path.dirname(save_path))
					torch.save(
						{
							'conds': conds,
							'conds_mean': conds_mean,
							'conds_std': conds_std,
						}, save_path)

		# evals those "fidXX"
		"""
		"fid<T>" = unconditional generation (conf.train_mode = diffusion).
			Note:   Diff. autoenc will still receive real images in this mode.
		"fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
			Note:   Diff. autoenc will still NOT receive real images in this made.
					but you need to make sure that the train_mode is latent_diffusion.
		"""
		for each in self.conf.eval_programs:
			if each.startswith('fid'):
				m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
				clip_latent_noise = False
				if m is not None:
					# eval(T1,T2)
					T = int(m[1])
					T_latent = int(m[2])
					print(f'evaluating FID T = {T}... latent T = {T_latent}')
				else:
					m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
					if m is not None:
						# fidclip(T1,T2)
						T = int(m[1])
						T_latent = int(m[2])
						clip_latent_noise = True
						print(
							f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
						)
					else:
						# evalT
						_, T = each.split('fid')
						T = int(T)
						T_latent = None
						print(f'evaluating FID T = {T}...')

				self.train_dataloader()
				sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
				if T_latent is not None:
					latent_sampler = self.conf._make_latent_diffusion_conf(
						T=T_latent).make_sampler()
				else:
					latent_sampler = None

				conf = self.conf.clone()
				conf.eval_num_images = 50_000
				score = evaluate_fid(
					sampler,
					self.ema_model,
					conf,
					device=self.device,
					train_data=self.train_data,
					val_data=self.val_data,
					latent_sampler=latent_sampler,
					conds_mean=self.conds_mean,
					conds_std=self.conds_std,
					remove_cache=False,
					clip_latent_noise=clip_latent_noise,
				)
				if T_latent is None:
					self.log(f'fid_ema_T{T}', score)
				else:
					name = 'fid'
					if clip_latent_noise:
						name += '_clip'
					name += f'_ema_T{T}_Tlatent{T_latent}'
					self.log(name, score)
		"""
		"recon<T>" = reconstruction & autoencoding (without noise inversion)
		"""
		for each in self.conf.eval_programs:
			if each.startswith('recon'):
				self.model: BeatGANsAutoencModel
				_, T = each.split('recon')
				T = int(T)
				print(f'evaluating reconstruction T = {T}...')

				sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

				conf = self.conf.clone()
				# eval whole val dataset
				conf.eval_num_images = len(self.val_data)
				# {'lpips', 'mse', 'ssim'}
				score = evaluate_lpips(sampler,
									self.ema_model,
									conf,
									device=self.device,
									val_data=self.val_data,
									latent_sampler=None)
				for k, v in score.items():
					self.log(f'{k}_ema_T{T}', v)
		"""
		"inv<T>" = reconstruction with noise inversion
		"""
		for each in self.conf.eval_programs:
			if each.startswith('inv'):
				self.model: BeatGANsAutoencModel
				_, T = each.split('inv')
				T = int(T)
				print(
					f'evaluating reconstruction with noise inversion T = {T}...'
				)

				sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

				conf = self.conf.clone()
				# eval whole val dataset
				conf.eval_num_images = len(self.val_data)
				# {'lpips', 'mse', 'ssim'}
				score = evaluate_lpips(sampler,
									self.ema_model,
									conf,
									device=self.device,
									val_data=self.val_data,
									latent_sampler=None,
									use_inverted_noise=True)
				for k, v in score.items():
					self.log(f'{k}_inv_ema_T{T}', v)


def ema(source, target, decay):
	source_dict = source.state_dict()
	target_dict = target.state_dict()
	for key in source_dict.keys():
		target_dict[key].data.copy_(target_dict[key].data * decay +
									source_dict[key].data * (1 - decay))


class WarmupLR:
	def __init__(self, warmup) -> None:
		self.warmup = warmup

	def __call__(self, step):
		return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
	closest = (num_samples // every) * every
	return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
	torch.set_num_threads(4)
	print('conf:', conf.name)
	# assert not (conf.fp16 and conf.grad_clip > 0
	#             ), 'pytorch lightning has bug with amp + gradient clipping'
	model = LitModel(conf)


	if not os.path.exists(conf.logdir):
		os.makedirs(conf.logdir)
	checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
								save_last=True,
								save_top_k=1,
								every_n_train_steps=conf.save_every_samples //
								conf.batch_size_effective)
	checkpoint_path = f'{conf.logdir}/last.ckpt'
	print('ckpt path:', checkpoint_path)
	# print(os.path.exists(checkpoint_path))
	# print(conf.continue_from)
	if os.path.exists(checkpoint_path):
		resume = checkpoint_path
		print('resume!')
	else:
		if conf.continue_from is not None:
			# continue from a checkpoint
			resume = conf.continue_from.path
		else:
			resume = None

	tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
											name=None,
											version='')

	# from pytorch_lightning.

	plugins = []
	if len(gpus) == 1 and nodes == 1:
		accelerator = None
	else:
		accelerator = 'ddp'
		from pytorch_lightning.plugins import DDPPlugin

		# important for working with gradient checkpoint
		plugins.append(DDPPlugin(find_unused_parameters=False))

	# print('resume : {}'.format(resume))
	trainer = pl.Trainer(
		max_steps=conf.total_samples // conf.batch_size_effective,
		resume_from_checkpoint=resume,
		gpus=gpus,
		num_nodes=nodes,
		accelerator=accelerator,
		precision=16 if conf.fp16 else 32,
		callbacks=[
			checkpoint,
			LearningRateMonitor(),
		],
		# clip in the model instead
		# gradient_clip_val=conf.grad_clip,
		replace_sampler_ddp=True,
		logger=tb_logger,
		accumulate_grad_batches=conf.accum_batches,
		plugins=plugins,
	)

	if mode == 'train':
		trainer.fit(model)
	elif mode == 'eval':
		# load the latest checkpoint
		# perform lpips
		# dummy loader to allow calling "test_step"
		# print('batch_size = {}'.format(conf.batch_size))
		# conf.batch_size = 16
		dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
						batch_size=conf.batch_size)
		eval_path = conf.eval_path or checkpoint_path
		# conf.eval_num_images = 50
		print('loading from:', eval_path)
		state = torch.load(eval_path, map_location='cpu')
		# print('step:', state['global_step'])
		# model.load_state_dict(state['state_dict'])
		model.load_state_dict(state)
		state['global_step'] = 55625

		# trainer.fit(model)
		out = trainer.test(model, dataloaders=dummy)
		# first (and only) loader
		out = out[0]
		print(out)

		if get_rank() == 0:
			# save to tensorboard
			for k, v in out.items():
				tb_logger.experiment.add_scalar(
					k, v, state['global_step'] * conf.batch_size_effective)

			# # save to file
			# # make it a dict of list
			# for k, v in out.items():
			#     out[k] = [v]
			tgt = f'evals/{conf.name}.txt'
			dirname = os.path.dirname(tgt)
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			with open(tgt, 'a') as f:
				f.write(json.dumps(out) + "\n")
			# pd.DataFrame(out).to_csv(tgt)
	else:
		raise NotImplementedError()
