# Large-scale image generation on ImageNet

Disentangled representations contain multiple interpretable and generative factors of data and capture separate factors of variation in the data. Such representations are more generalizable, robust against adversarial attacks, and lead to better image generation. Variational Autoencoders (VAE) are commonly used for disentangling independent factors from observed data such that single generative factors are influenced by change in a single latent unit. But scaling the disentanglement performance to a large scale dataset like ImageNet is still a challenge. In this project, we  improve image generation, model convergence, and data efficiency of the current state-of-the-art model for image generation on ImageNet, DenseFlow. We also investigate if an auxiliary loss function, such as the contrastive center-loss can induce better geometric class separability in the encoding. The resulting VAE model with disentangled representations could potentially be used as a basis for transfer learning for other image generation or classification tasks.

## Setup data for training and evaluation
Run setup_data_imagenet32.sh to download and process the data 

## Training

```
cd ./experiments/image
```

### ImageNet32
```
python train.py --epochs 20 --batch_size 32 --optimizer adamax --lr 1e-3  --gamma 0.95 --warmup 5000  --eval_every 1 --check_every 1 --dataset imagenet32 --augmentation eta --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 24 24 24 --growth_rate 10  --name DF_74_10
```

```
python train_more.py --model ./log/imagenet32_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 20
```

### CIFAR10
```
python train.py --epochs 32 --batch_size 64 --optimizer adamax --lr 1e-3  --gamma 0.9975 --warmup 5000  --eval_every 1 --check_every 10 --dataset cifar10 --augmentation eta --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_74_10
```

```
python train_more.py --model ./log/cifar10_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 32
```

## Evaluation
```
python eval_loglik.py --model PATH_TO_MODEL --k 200 --kbs 50
```

## Image Generation
```
python eval_sample.py --model PATH_TO_MODEL
```


## References
1. Matej Grcic, Ivan Grubisic, and Sinisa Segvic. “Densely connected normalizing flows”. In: CoRR abs/2106.04627 (2021). arXiv: [2106.04627](https://arxiv.org/abs/2106.04627)

2. Rewon Child. “Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images”. In: CoRR abs/2011.10650 (2020). arXiv: [2011.10650](https://arxiv.org/abs/2011.10650)


## Acknowledgement
Our project builds on [DenseFlow](https://github.com/matejgrcic/DenseFlow). We thank the authors for releasing their code.

