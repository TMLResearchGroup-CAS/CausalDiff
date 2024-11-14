# # coding:utf8
# from __future__ import print_function
# import torch.optim as optim
# from torch import nn, optim, autograd
# # from utils import *
# # from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
# # from models import *
# import torch.nn.functional as F
# import torch.nn as nn
# import sys
# from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.datasets import CIFAR10 as CIFAR10_torch
# import random
# from autoattack import AutoAttack
# import numpy as np
# from torch.utils.data import DataLoader, SubsetRandomSampler
# import matplotlib.pyplot as plt
# import copy
import torch
import random
from templates import *
import matplotlib.pyplot as plt
from autoattack import AutoAttack

# nohup python -u d_LaCIM.py --epochs 20 --optimizer sgd --lr 0.3 --lr_decay 0.5 --lr_controler 120 --in_channel 3 --batch-size 256 --test-batch-size 256 --reg 0.0005 --dataset mnist --num_classes 2 --env_num 2 --seed -1 --zs_dim 32 --root ./data/colored_MNIST_0.02_env_2_0_c_2_0.10/ --test_ep 10 --lr2 0.007 --reg2 0.08 --sample_num 10 --image_size 28 --alpha 8.0 --gamma 1.0 --beta 1.0 --z_ratio 0.5 > run_d_baseline_checkpoint1.log 2>&1 &
# nohup python -u d_LaCIM.py --epochs 20 --optimizer sgd --lr 0.3 --lr_decay 0.5 --lr_controler 120 --in_channel 3 --batch-size 256 --test-batch-size 256 --reg 0.0005 --dataset mnist --num_classes 2 --env_num 2 --seed -1 --zs_dim 32 --root ./data/colored_MNIST_0.02_env_2_0_c_2_0.10/ --test_ep 10 --lr2 0.007 --reg2 0.08 --sample_num 10 --image_size 28 --alpha 8.0 --gamma 1.0 --beta 1.0 --z_ratio 0.5 > run_d_baseline_checkpoint_unnorm.log 2>&1 &
# nohup python -u d_LaCIM.py --epochs 20 --optimizer sgd --lr 0.3 --lr_decay 0.5 --lr_controler 120 --in_channel 3 --batch-size 256 --test-batch-size 256 --reg 0.0005 --dataset mnist --num_classes 2 --env_num 2 --seed -1 --zs_dim 32 --root ./data/colored_MNIST_2/ --test_ep 10 --lr2 0.007 --reg2 0.08 --sample_num 10 --image_size 28 --alpha 8.0 --gamma 1.0 --beta 1.0 --z_ratio 0.5 > run_CMNIST_0_7.log 2>&1 &

# nohup python -u d_LaCIM.py --epochs 20 --optimizer sgd --lr 0.3 --lr_decay 0.5 --lr_controler 120 --in_channel 3 --batch-size 256 --test-batch-size 256 --reg 0.0005 --dataset mnist --num_classes 2 --env_num 2 --seed -1 --zs_dim 128 --root ./data/data_decorrelated_shuffle/ --test_ep 100 --lr2 0.007 --reg2 0.08 --sample_num 10 --image_size 28 --alpha 8.0 --gamma 1.0 --beta 1.0 --z_ratio 0.5 --eval_mode x > run_d_decorrelated_shuffle_eval_mode_x_pgd100_alpha0001_clamp.log 2>&1 &


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def mean_nll(logits, y):
	return F.nll_loss(torch.log(logits), y)


def t_sample(T, batch_size, device):
	w = np.ones([T])
	p = w / np.sum(w)
	indices_np = np.random.choice(len(p), size=(batch_size, ), p=p)
	indices = torch.from_numpy(indices_np).long().to(device)
	weights_np = 1 / (len(p) * p[indices_np])
	weights = torch.from_numpy(weights_np).float().to(device)
	return indices, weights


def _extract_into_tensor(arr, timesteps, broadcast_shape):
	"""
	Extract values from a 1-D numpy array for a batch of indices.

	:param arr: the 1-D numpy array.
	:param timesteps: a tensor of indices into the array to extract.
	:param broadcast_shape: a larger shape of K dimensions with the batch
							dimension equal to the length of timesteps.
	:return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
	"""
	res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
	while len(res.shape) < len(broadcast_shape):
		res = res[..., None]
	return res.expand(broadcast_shape)


def update_state_dict(state_dict, idx_start=9):

	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		if 'dense' in k:
			continue
		name = k[idx_start:]  # remove 'module.0.' of dataparallel
		new_state_dict[name]=v

	return new_state_dict

def _extract_into_tensor(arr, timesteps, broadcast_shape):
	res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
	while len(res.shape) < len(broadcast_shape):
		res = res[..., None]
	return res.expand(broadcast_shape)

def main():
	# args = get_opt()
	
	# args = make_dirs(args)
	# logger = get_logger(args)
	# logger.info(str(args))
	# args.logger = logger
	# other_info = {}
	
	# if args.seed != -1:
	# 	torch.manual_seed(args.seed)
	# 	if args.cuda:
	# 		torch.cuda.manual_seed(args.seed)

	seed = 0
	print('seed = {}'.format(seed))
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 						std=[0.229, 0.224, 0.225])
	# unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

	# train_data = CIFAR10(download=True,root='/data/users/zhangmingkun//DiffPure-master/dataset', transform=transforms.Compose([
	# 	# transforms.RandomHorizontalFlip(),
	# 	# transforms.RandomCrop(32, 4),
	# 	transforms.ToTensor(),
	# 	# normalize,
	# 	]))
	test_data = CIFAR10_torch(root='/data/users/zhangmingkun//DiffPure-master/dataset', train=False, transform=transforms.Compose([
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomCrop(32, 4),
		transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		# normalize,
		]))
	# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
	test_loader = DataLoader(test_data, batch_size=10, shuffle=False, pin_memory=True)

	# device = 'cuda:1'
	conf = cifar10_autoenc()
	# print(conf.name)
	model = LitModel(conf)
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencencode/last.ckpt', map_location='cpu')
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode/last.ckpt', map_location='cpu')
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/ckpt_9.pth.tar', map_location='cpu')
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode_ckpt/ckpt/45000.ckpt', map_location='cpu')
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode_ckpt/ckpt/68125_ema_eval.ckpt', map_location='cpu')
	# state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200/ckpt/50625_ema_eval.ckpt', map_location='cpu')
	state = torch.load('/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold_scale03/ckpt/40000_ema.ckpt', map_location='cpu')

	# model.load_state_dict(state['state_dict'], strict=False)
	model.load_state_dict(state)
	# model.to(device)
	# model.ema_model.eval()
	model.ema_model.to(device)
	# model.ema_model.requires_grad = True

	# attack_list = ['apgd-ce']
	# attack_lp = 'L2'
	# attack_eps = 0.5
	# adversary_cls = AutoAttack(model.inference_adv, norm=attack_lp, eps=attack_eps,
	# 							version='custom', attacks_to_run=attack_list,
	# 							log_path=f'log_cls_train.txt', device=device)

	# x_val, y_val = next(iter(test_loader))
	# x_val, y_val = x_val.to(device), y_val.to(device)
	# x_adv, acc, rob = adversary_cls.run_standard_evaluation(x_val, y_val, bs=10)
	# print('acc = {}, rob = {}'.format(acc, rob))
	# torch.save({'x_val': x_val, 'x_adv': x_adv, 'y_val': y_val}, 'res_adv.pt')

	# x_val = torch.load('res_adv_E_all_loss_64.pt')['x_val']
	x_adv = torch.load('res_adv_E_all_loss_64.pt')['x_adv']

	res = torch.load('res_adv_E_all_loss_cond_and_uncond.pt')
	x_val = res['x_val'][:10]

	# res = torch.load('res_adv_10.pt')
	# x_val = res['x_val']
	# x_adv = res['x_adv']
	# y_val = res['y_val']
	# print('y_val: ')
	# print(y_val)

	x_val, x_adv = x_val.to(device), x_adv.to(device)
	# x_adv = x_adv.to(device)

	x_val = (x_val - 0.5) * 2
	x_adv = (x_adv - 0.5) * 2
	# plot x_adv
	cond = model.search_rv(x_val.detach())
	# cond_adv = model.search_rv_adv(x_adv.detach())
	# torch.save({'cond': cond, 'cond_adv': cond_adv}, 'res_cond.pt')

	# cond_res = torch.load('res_cond.pt')
	# cond = cond_res['cond'].to(device)
	# cond_adv = cond_res['cond_adv'].to(device)
	
	xT = model.encode_stochastic(x_val, cond, T=20)
	torch.save(xT, 'xT.pt')
	assert i == 3
	xT_adv = model.encode_stochastic(x_adv, cond_adv, T=20)


	pred_img_zs = model.render(xT, cond, T=20)
	pred_img_zs_adv = model.render(xT_adv, cond_adv, T=20)

	x_val = (x_val + 1.) / 2
	x_adv = (x_adv + 1.) / 2

	z_adv = cond_adv[:, :256]
	s_adv = cond_adv[:, 256:]
	z_masked = torch.zeros_like(z_adv)
	s_masked = torch.zeros_like(s_adv)
	z_sm_adv = torch.cat([z_adv, s_masked], dim=1)
	zm_s_adv = torch.cat([z_masked, s_adv], dim=1)

	pred_img_z_sm_adv = model.render(xT_adv, z_sm_adv, T=20)
	pred_img_zm_s_adv = model.render(xT_adv, zm_s_adv, T=20)

	pred_img_zs_masked_zeros = model.render(xT, torch.zeros_like(cond_adv), T=20)

	# pred_img_zs_masked_randn = model.render(xT, torch.randn_like(cond), T=20)

	pred_y_masked_z = model.inference((pred_img_zm_s_adv - 0.5) * 2).detach().cpu()
	print('pred_y of masked z : ')
	print(np.argmax(pred_y_masked_z, axis=1))
	

	# pred_img_zs = (pred_img_zs + 1) / 2
	# pred_img_zs_masked = (pred_img_zs_masked + 1) / 2

	plt.figure(figsize=(25, 10))
	# label = ['x_adv by AdpAttack', 'xT=encode_stochastic(x)', 'recon\n(cond=semantic latent)', 'recon\n(cond=zeros)', 'recon\n(cond=randn)', 'recon \n (cond=[z, masked s])', 'recon \n (cond=[masked z,s])']
	label = ['x', 'recon\n(cond=zs)', 'recon\n(cond=zeros)', 'x_adv by AdpAttack', 'recon\n(cond=zs_adv)', 'recon \n (cond=[masked z,s_adv])', 'recon\n(cond=[z_adv, maksed s])']
	# label = ['x', 'xT=zeros', 'recon\n(cond=semantic latent)', 'recon\n(cond=zeros)']
	for i, data in enumerate([x_val, pred_img_zs, pred_img_zs_masked_zeros, x_adv, pred_img_zs_adv, pred_img_zm_s_adv, pred_img_z_sm_adv]):
		for j, img in enumerate(data):
			# if i != 0:
				# data = (data.detach().clamp(-1., 1.) + 1) / 2
			# img = (img.clamp(-1., 1.) + 1)/2
			plt.subplot(7, 10, i*10 + j + 1)
			plt.imshow((img).detach().permute(1, 2, 0).cpu())
			plt.axis('off') 

			if j == 0:
				plt.text(-32*2+10, 32/2, label[i], ha='center', va='center', fontsize=12)

	plt.savefig('cond_gen_adv_E_all_loss.png')
	print('fig saved !!!')


	# print('acc = {}, rob = {}'.format(acc, rob))

	model.cpu()


if __name__ =='__main__':
	main()