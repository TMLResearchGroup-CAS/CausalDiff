import torch

# path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode/last.ckpt'
# path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencprior01_b_1e_2_g_1e_2_pretrain_encode/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode_woema/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_woprior_b_1e_2_g_1e_2_pretrain_encode/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode_ckpt/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_encode_ckpt/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_entropy_by_z/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_epoch1442_entropy_by_z/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_epoch1442/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_b_1e_2_g_1e_2_pretrain_epoch1442/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_l2_pretrain_epoch200_norm/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/run_cifar10_disentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold04/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold_scale/last.ckpt'

path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold_scale04/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_01_pretrain_epoch200_norm_mask_threshold_scale/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_l2_01_pretrain_epoch200_norm_mask_threshold_scale/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_01_pretrain_epoch200_norm_mask_threshold_scale03_zeros/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold_scale01/last.ckpt'
path = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoencdisentangle_zs_prior01_club_entropy_cos_pretrain_epoch200_norm_mask_threshold_3e_2_scale01/last.ckpt'

path_encode = '/data/users/zhangmingkun/diffae_causal/checkpoints_old/cifar10_autoencencode_200/last.ckpt'

ckpt = torch.load(path, map_location='cpu')
ckpt_encode = torch.load(path_encode, map_location='cpu')
ckpt['state_dict'] = ckpt_encode['state_dict']
torch.save(ckpt, path)


path_latent = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoenc_latent_latent_cos_pretrain_epoch200_norm_mask_threshold_scale01_new_27500/last.ckpt'
path_ckpt = '/data/users/zhangmingkun/diffae_causal/checkpoints/cifar10_autoenc_latent_latent_cos_pretrain_epoch200_norm_mask_threshold_scale01_new_27500/27500.ckpt'

ckpt_latent = torch.load(path_latent, map_location='cpu')
ckpt_ckpt = torch.load(path_ckpt, map_location='cpu')
ckpt_latent['state_dict'] = ckpt_ckpt
torch.save(ckpt_latent, path_latent)