# Large-scale image generation on ImageNet

This project seeks to develop a VAE model that can learn disentangled latent representations of data, thus capturing class information alongside the underlying generative factors of the data. The hypothesis is that disentangled latent representations will improve model convergence rates and the generation quality of the decoder network. 

DenseFlow, proposed in [Densely Connected Normalizing Flows](https://arxiv.org/abs/2106.04627) will be adapted using an auxiliary loss function, such as the contrastive centerloss to induce geometric class separability in the encoding. This would result in better class-conditional image generation and also enable robust classification with simple classifiers. The resulting VAE model with disentangled representations could potentially be used as a basis for transfer learning for other image classification or generation tasks.


# Acknowledgement
Our project builds on [DenseFlow](https://github.com/matejgrcic/DenseFlow). We thank the authors for releasing their code.

