# Large-scale image generation on CIFAR-10

Training of the DenseFlow model is quite time consuming - 250 hours to train 400 epochs on the complete CIFAR-10 dataset using a single RTX 3090 GPU can achieved a bpd of 2.98

Adapting ideas from the [Very Deep VAE](https://arxiv.org/pdf/2011.10650.pdf), we experimented in expanding layers on DenseFlow to investigate if deeper architectures would lead to similar or better performance with faster convergence.

For the expand on layers experiment in DenseFlow model, we expanded original DenseFlow blocksfrom [6, 4, 1] to [8, 6, 2] to increase the depth and complexity of DenseFlow architecture.

 Although the training overhead increased from 3400s/epoch to 4400s/epoch, we achieved a much faster convergence rate. We reduced the training time to 1/6 of original, and achieves comparable image generation quality with a bpd of 3.17


##  Setup

- CUDA 11.1
- Python 3.8

```
pip install -r requirements.txt
pip install -e .
```
## Training

```
cd ./experiments/image
```
CIFAR-10:
```
python train.py --epochs 24 --batch_size 32 --optimizer adamax --lr 1e-3  --gamma 0.9975 --warmup 5000  --eval_every 1 --check_every 1 --dataset cifar10 --augmentation eta --block_conf 8 6 2 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_DEEP
```
```
python train_more.py --model ./log/cifar10_8bit/densenet-flow/expdecay/DF_DEEP --new_lr 2e-5 --new_epochs 32
```

## Evaluation

CIFAR-10:
```
python eval_loglik.py --model PATH_TO_MODEL --k 1000 --kbs 50
```

## Model weights
Final Model weights after 32 epochs are stored [here](https://drive.google.com/file/d/1-Y5yI617CPMrIzcKBf3-tQl16Qnlvesk/view?usp=sharing).

## Samples generation
Generated samples are stored in `PATH_TO_MODEL/samples`
```
python eval_sample.py --model PATH_TO_MODEL
```
**Note:** `PATH_TO_MODEL` has to contain `check` directory.

### CIFAR-10 at epoch 1

![Alt text](assets/sample_ep1_s0.png?raw=true)

### CIFAR-10 at epoch 32

![Alt text](assets/sample_ep32_s0.png?raw=true)

### Train_valid_bpd_curve

![Alt text](assets/train_valid_bpd_curve.png?raw=true)

Details records can be find in [assets](https://github.com/meghana17/11785-Project-VAE/tree/dev/assets)

## References

Rewon Child. “Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images”. In: CoRR abs/2011.10650 (2020). arXiv: [2011.10650](https://arxiv.org/abs/2011.10650)

## Acknowledgement
Our project builds on [DenseFlow](https://github.com/matejgrcic/DenseFlow). We thank the authors for releasing their code.
