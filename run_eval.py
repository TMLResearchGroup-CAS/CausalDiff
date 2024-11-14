# coding:utf8
from torchvision import transforms
from torchvision.datasets import CIFAR10 as CIFAR10_torch
from autoattack import AutoAttack
import torch
import random
from templates import *
import matplotlib.pyplot as plt
from bpda_eot.bpda_eot_attack import BPDA_EOT_Attack
import time


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

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
	conf = cifar10_autoenc()
	seed = 0
	print('seed = {}'.format(seed))
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	test_data = CIFAR10_torch(root='./dataset', train=False, transform=transforms.Compose([
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomCrop(32, 4),
		transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		# normalize,
		]))

	index = 0
	num = 128
	begin = num * index
	end = num * index + num
	sampled_data = [item for i, item in enumerate(test_data) if begin <= i < end]
	test_loader = DataLoader(sampled_data, batch_size=len(sampled_data), shuffle=False, pin_memory=True)

	
	model = LitModel(conf)
for i in range(1):
	state = torch.load('./checkpoints/cifar10_autoenc_prior01_pretrain_threshold_zeros01_decouple_zbias_cls_mizs1e_5/last.ckpt', map_location='cpu')


	if 'state_dict' in state.keys():
		model.load_state_dict(state['state_dict'], strict=False)
	else:
		model.load_state_dict(state, strict=False)
	# model.load_state_dict(state)
	
	model.cls = WideResNet(depth=70, widen_factor=16, dropRate=0.3)
	state = torch.load('/home/users/zhangmingkun/diffae_causal/weights.pt')

	# self.cls = RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75],
	# 		override_groups_map=None, num_classes=100)
	# self.cls.load_state_dict(torch.load("/home/users/zhangmingkun/diffae_causal_17/cifar100_repvgg.pt"))

	# self.cls = WideResNet(depth=40, widen_factor=2, num_classes=100)
	# # self.cls.load_state_dict(torch.load("/home/users/zhangmingkun/RDC/resources/checkpoints/cifar100/wrn_40_2.pth"))
	# self.cls.load_state_dict(torch.load("/home/users/zhangmingkun/diffae_causal/wrn_40_2.pth")["model"])
	
	r = {}
	for k, v in list(state.items()):
		k = k.split('module.', 1)[1]
		r[k] = v
	model.cls.load_state_dict(r)

	model.eval()
	model.to(device)
	# model.ema_model.train()
	model.ema_model.eval()
	model.ema_model.to(device)

	model.zero_grad()

	lr_E = 0.1
	for lr in [1e5]:
		print('lr_E = {}, lr = {}'.format(lr_E, lr))
		model.conf.lr_search_E = lr_E
		model.conf.lr_search = lr

		# !!! standard mode
		attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
		attack_lp = 'Linf'
		attack_eps = 8./255.
		# attack_lp = 'L2'
		# attack_eps = 0.5

		# adversary = AutoAttack(model.inference_cls, norm=attack_lp, eps=attack_eps, version='standard', attacks_to_run=attack_list, log_path=f'log_cls_train.txt', device=device)
		adversary = AutoAttack(model.inference_causal_purify, norm=attack_lp, eps=attack_eps, version='custom', attacks_to_run=attack_list, log_path=f'log_cls_train.txt', device=device)
		# adversary = AutoAttack(model.inference_causal, norm=attack_lp, eps=attack_eps, version='custom', attacks_to_run=attack_list, log_path=f'log_cls_train.txt', device=device)
		
		x_val, y_val = next(iter(test_loader))
		x_adv, acc, rob = adversary.run_standard_evaluation(x_val, y_val, bs=20)
		# results = {'x_val': x_val, 'x_adv': x_adv, 'y_val': y_val}
		# torch.save(results, 'res_adv_unbound.pt')
		# torch.save(results, 'res_adv_mizs1e_5_cls_only_512.pt'.format(lr_E, lr))


		# # !!! rand mode
		# attack_list = ['apgd-ce', 'apgd-dlr']
		# attack_lp = 'Linf'
		# attack_eps = 8/255.
		# # attack_lp = 'L2'
		# # attack_eps = 0.5
		
		# adversary = AutoAttack(model.inference_cls, norm=attack_lp, eps=attack_eps, version='rand', attacks_to_run=attack_list, log_path=f'log_cls_train.txt', device=device)
		# adversary.apgd.eot_iter = 20
		# x_val, y_val = next(iter(test_loader))
		# x_adv, acc, rob = adversary.run_standard_evaluation(x_val, y_val, bs=20)

		# print('results with lr_E = {}, lr = {} saved!'.format(lr_E, lr))

		# # !!! BPDA
		# attack_eps = 8/255.
		# x_val, y_val = next(iter(test_loader))
		# x_val = x_val.to(device)
		# y_val = y_val.to(device)
		# adversary = BPDA_EOT_Attack(model.inference_cls, adv_eps=attack_eps, eot_defense_reps=20, eot_attack_reps=15)

		# start_time = time.time()
		# # model_.reset_counter()
		# # model_.set_tag()
		# class_batch, ims_adv_batch = adversary.attack_all(x_val, y_val, batch_size=4)
		# init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
		# robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]

		# print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))



if __name__ =='__main__':
	main()