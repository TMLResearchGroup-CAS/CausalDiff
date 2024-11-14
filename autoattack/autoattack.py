import math
import time

import numpy as np
import torch

from .other_utils import Logger
from autoattack import checks
import random
from scipy.ndimage.interpolation import rotate as scipyrotate
import matplotlib.pyplot as plt


def patch_mask(x, image_size, channel, patch_w, patch_h, pos_w, pos_h):
	
	mask = torch.ones((1, channel, image_size, image_size))
	if channel == 1:
		for i in range(patch_w):
			if pos_w+i >= image_size:
				break
			mask[0][0][pos_w+i][pos_h:min(pos_h+patch_h, image_size)] = 0.
			mask = mask.repeat((x.size(0), 1, 1, 1))
			return mask, torch.mul(mask, x)

	elif channel == 3:
		patch = torch.randn((channel, patch_w, patch_h))
		for t in range(x.size(0)):
			for c in range(channel):
				for i in range(patch_w):
					if pos_w+i >= image_size:
						break
					mask[0][c][pos_w+i][pos_h:min(pos_h+patch_h, image_size)] = patch[c][i][:]
					x[t][c][pos_w+i][pos_h:min(pos_h+patch_h, image_size)] = patch[c][i][:]

		return mask, x

rotate=45
def rotation(images, device):
	shape = images.shape
	mean = []
	for c in range(shape[1]):
		mean.append(float(torch.mean(images[:,c])))
	for i in range(shape[0]):
		im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
		r = int((im_.shape[-2] - shape[-2]) / 2)
		c = int((im_.shape[-1] - shape[-1]) / 2)
		images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)
	return images

class AutoAttack():
	def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
				attacks_to_run=[], version='standard', is_tf_model=False,
				device='cuda', log_path=None):
		self.model = model
		self.norm = norm
		assert norm in ['Linf', 'L2', 'L1']
		self.epsilon = eps
		self.seed = seed
		self.verbose = verbose
		self.attacks_to_run = attacks_to_run
		self.version = version
		self.is_tf_model = is_tf_model
		self.device = device
		self.logger = Logger(log_path)

		# if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
		#     raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
		
		if not self.is_tf_model:
			from .autopgd_base import APGDAttack
			self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
				eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
				device=self.device, logger=self.logger)
			
			from .fab_pt import FABAttack_PT
			self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
				norm=self.norm, verbose=False, device=self.device)
		
			from .square import SquareAttack
			self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
				n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
				
			from .autopgd_base import APGDAttack_targeted
			self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
				eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
				logger=self.logger)
	
		else:
			from .autopgd_base import APGDAttack
			self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
				eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
				is_tf_model=True, logger=self.logger)
			
			from .fab_tf import FABAttack_TF
			self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
				norm=self.norm, verbose=False, device=self.device)
		
			from .square import SquareAttack
			self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
				n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
				
			from .autopgd_base import APGDAttack_targeted
			self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
				eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
				is_tf_model=True, logger=self.logger)
	
		if version in ['standard', 'plus', 'rand']:
			self.set_version(version)
		
	def get_logits(self, x, req_grad=True):
		if not self.is_tf_model:
			# print(self.model.module.device)
			# print(x.device)
			return self.model(x, req_grad)
		else:
			return self.model.predict(x)
	
	def get_seed(self):
		return time.time() if self.seed is None else self.seed
	
	def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False):
		# print('in run_standard_evaluation !')
		if self.verbose:
			print('using {} version including {}'.format(self.version,
				', '.join(self.attacks_to_run)))
		
		# checks on type of defense
		if self.version != 'rand':
			checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
				y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
		n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
			logger=self.logger)
		checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
			logger=self.logger)
		checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
			self.fab.n_target_classes, logger=self.logger)
		
		print('start clean accuracy evaluation!')
		with torch.no_grad():
			# calculate accuracy
			n_batches = int(np.ceil(x_orig.shape[0] / bs))
			robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
			y_adv = torch.empty_like(y_orig)
			for batch_idx in range(n_batches):
				start_idx = batch_idx * bs
				end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

				x = x_orig[start_idx:end_idx, :].clone().to(self.device)
				
				# random patch
				# patch_w = random.choice([i for i in range(5,15)])
				# patch_h = 20 - patch_w
				# pos_w = random.choice([i for i in range(5, 20)])
				# pos_h = random.choice([i for i in range(5, 20)])
				# _, x = patch_mask(x, 32, 3, patch_w, patch_h, pos_w, pos_h)
				# x = x.to(self.device)
				# random patch modify over here!

				# 45 degree rotation
				# x = rotation(x, self.device)
				# 45 degree rotation modify over here!

				y = y_orig[start_idx:end_idx].clone().to(self.device)
				output = self.get_logits(x, req_grad=False).max(dim=1)[1]
				y_adv[start_idx: end_idx] = output
				correct_batch = y.eq(output)
				robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

			robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
			robust_accuracy_dict = {'clean': robust_accuracy}
			clean_acc = robust_accuracy
			
			if self.verbose:
				self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
				# self.logger.log('random patch accuracy: {:.2%}'.format(robust_accuracy))
			
			# return 
			print('start robust evaluation!')
			x_adv = x_orig.clone().detach()
			startt = time.time()
			for attack in self.attacks_to_run:
				print('start ' + str(attack) + ' attack!')
				# item() is super important as pytorch int division uses floor rounding
				num_robust = torch.sum(robust_flags).item()

				if num_robust == 0:
					break

				n_batches = int(np.ceil(num_robust / bs))

				robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
				if num_robust > 1:
					robust_lin_idcs.squeeze_()
				
				for batch_idx in range(n_batches):
					start_idx = batch_idx * bs
					end_idx = min((batch_idx + 1) * bs, num_robust)

					batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
					if len(batch_datapoint_idcs.shape) > 1:
						batch_datapoint_idcs.squeeze_(-1)
					x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
					y = y_orig[batch_datapoint_idcs].clone().to(self.device)

					# make sure that x is a 4d tensor even if there is only a single datapoint left
					if len(x.shape) == 3:
						x.unsqueeze_(dim=0)
					
					# run attack
					if attack == 'apgd-ce':
						# apgd on cross-entropy loss
						self.apgd.loss = 'ce'
						self.apgd.seed = self.get_seed()
						adv_curr = self.apgd.perturb(x, y) #cheap=True
					
					elif attack == 'apgd-dlr':
						# apgd on dlr loss
						self.apgd.loss = 'dlr'
						self.apgd.seed = self.get_seed()
						adv_curr = self.apgd.perturb(x, y) #cheap=True
					
					elif attack == 'fab':
						# fab
						self.fab.targeted = False
						self.fab.seed = self.get_seed()
						adv_curr = self.fab.perturb(x, y)
					
					elif attack == 'square':
						# square
						self.square.seed = self.get_seed()
						adv_curr = self.square.perturb(x, y)
					
					elif attack == 'apgd-t':
						# targeted apgd
						self.apgd_targeted.seed = self.get_seed()
						adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
					
					elif attack == 'fab-t':
						# fab targeted
						self.fab.targeted = True
						self.fab.n_restarts = 1
						self.fab.seed = self.get_seed()
						adv_curr = self.fab.perturb(x, y)
					
					else:
						raise ValueError('Attack not supported')
				
					# print(adv_curr.size())
					# plt.figure(figsize=(20, 16)) 
					# for i in range(10):
					# # while True:
					# 	plt.subplot2grid((10,10), (0,i))
					# 	plt.imshow(np.asarray((255*x[i]).detach().cpu().numpy().astype('uint8').transpose((1,2,0))),cmap='Greys_r', interpolation=None)
					# 	if i == 0:
					# 		plt.ylabel('x')#,rotation=0)

					# 	plt.xticks([])
					# 	plt.yticks([])
					# 	plt.subplot2grid((10,10), (1,i))
					# 	plt.imshow(np.asarray((255*adv_curr[i]).detach().cpu().numpy().astype('uint8').transpose((1,2,0))),cmap='Greys_r', interpolation=None)
					# 	if i == 0:
					# 		plt.ylabel('x_adv')#,rotation=0)
							
					# 	plt.xticks([])
					# 	plt.yticks([])
					# 	# plt.xlabel(str((BCE_list[i])))
					# plt.savefig('adv_curr.png')
					
					# torch.save([x, adv_curr], 'x_adv.pt')
					# assert i ==1
					output = self.get_logits(adv_curr).max(dim=1)[1]
					# assert i == 1
					false_batch = ~y.eq(output).to(robust_flags.device)
					non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
					robust_flags[non_robust_lin_idcs] = False

					x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
					y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

					if self.verbose:
						num_non_robust_batch = torch.sum(false_batch)    
						self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
							attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
				
				robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
				robust_accuracy_dict[attack] = robust_accuracy
				if self.verbose:
					self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
						attack.upper(), robust_accuracy, time.time() - startt))
					
			# check about square
			checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
			
			# final check
			if self.verbose:
				if self.norm == 'Linf':
					res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
				elif self.norm == 'L2':
					res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
				elif self.norm == 'L1':
					res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
				self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
					self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
				# self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
		x_adv = adv_curr
		if return_labels:
			return x_adv, y_adv
		else:
			return x_adv, clean_acc, robust_accuracy
		
	def clean_accuracy(self, x_orig, y_orig, bs=250):
		n_batches = math.ceil(x_orig.shape[0] / bs)
		acc = 0.
		for counter in range(n_batches):
			x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
			y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
			output = self.get_logits(x)
			acc += (output.max(1)[1] == y).float().sum()
			
		if self.verbose:
			print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
		
		return acc.item() / x_orig.shape[0]
		
	def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
		if self.verbose:
			print('using {} version including {}'.format(self.version,
				', '.join(self.attacks_to_run)))
		
		l_attacks = self.attacks_to_run
		adv = {}
		verbose_indiv = self.verbose
		self.verbose = False
		
		for c in l_attacks:
			startt = time.time()
			self.attacks_to_run = [c]
			x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
			if return_labels:
				adv[c] = (x_adv, y_adv)
			else:
				adv[c] = x_adv
			if verbose_indiv:    
				acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
				space = '\t \t' if c == 'fab' else '\t'
				self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
					c.upper(), space, acc_indiv,  time.time() - startt))
		
		return adv
		
	def set_version(self, version='standard'):
		if self.verbose:
			print('setting parameters for {} version'.format(version))
		
		if version == 'standard':
			self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
			if self.norm in ['Linf', 'L2']:
				self.apgd.n_restarts = 1
				self.apgd_targeted.n_target_classes = 9
			elif self.norm in ['L1']:
				self.apgd.use_largereps = True
				self.apgd_targeted.use_largereps = True
				self.apgd.n_restarts = 5
				self.apgd_targeted.n_target_classes = 5
			self.fab.n_restarts = 1
			self.apgd_targeted.n_restarts = 1
			self.fab.n_target_classes = 9
			#self.apgd_targeted.n_target_classes = 9
			self.square.n_queries = 5000
		
		elif version == 'plus':
			self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
			self.apgd.n_restarts = 5
			self.fab.n_restarts = 5
			self.apgd_targeted.n_restarts = 1
			self.fab.n_target_classes = 9
			self.apgd_targeted.n_target_classes = 9
			self.square.n_queries = 5000
			if not self.norm in ['Linf', 'L2']:
				print('"{}" version is used with {} norm: please check'.format(
					version, self.norm))
		
		elif version == 'rand':
			self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
			self.apgd.n_restarts = 1
			self.apgd.eot_iter = 20

