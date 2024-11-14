from templates import *
from templates_latent import *

if __name__ == '__main__':
	gpus = [0]
	conf = cifar10_autoenc()
	train(conf, gpus=gpus)