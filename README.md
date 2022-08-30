# hmnist_classification

## Dataset

[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): a large collection of multi-source dermatoscopic images of pigmented lesions (Downloaded in the Kaggle)

## ViT related materials:

[Vision Transformer(paperswithcode)-- refer to this website for more SOTA models](https://paperswithcode.com/method/vision-transformer): The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.
    
### Useful PyTorch Implementation of ViT:(Remember to reference it if written on a paper)

  - ***[timm implementation](https://github.com/rwightman/pytorch-image-models)***:
    - About: PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet 
    - For more refer to the ***[timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)***:
      - Py**T**orch **Im**age **M**odels (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results. -- All model architecture families include variants with pretrained weights.
      - timm's vit source code: [pytorch-image-models/timm/models/vision_transformer.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
      - To obtain reference code and pretrained weights, see [Google's repository for the ViT model](https://github.com/google-research/vision_transformer)

  - ***[lucidrains implementation](https://github.com/lucidrains/vit-pytorch)***:

    - About: Implementation of Vision Transformer, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Pytorch.
    - Significance is further explained in [Yannic Kilcher's](https://www.youtube.com/watch?v=TrdevFK_am4) video.
    - For a Pytorch implementation with pretrained models, please see Ross Wightman's repository [here(timm implementation)](https://github.com/rwightman/pytorch-image-models)
    - The official Jax repository is [here(Google's repository for the ViT model)](https://github.com/google-research/vision_transformer).

  - ***[Facebook official DeiT repository](https://github.com/facebookresearch/deit)***: 
    include models:
    - DeiT Data-Efficient Image Transformers, ICML 2021 [bib]
    - CaiT (Going deeper with Image Transformers), ICCV 2021 [bib]
    - ResMLP (ResMLP: Feedforward networks for image classification with data-efficient training) [bib]
    - PatchConvnet (Augmenting Convolutional networks with attention-based aggregation) [bib]
    - 3Things (Three things everyone should know about Vision Transformers), ECCV 2022 [bib]
    - DeiT III (DeiT III: Revenge of the ViT), ECCV 2022 [**bib**]
  
  - ***[jankrepl/mildlyoverfitted](https://github.com/jankrepl/mildlyoverfitted)***: 
    - mildlyoverfitted's customized implementation of ViT
    - the GitHub containing paper implementations from scratch and machine learning tutorials. 
    - Here is the corresponding [YouTube video custom ViT implementation](https://www.youtube.com/watch?v=ovB0ddFtzzA&ab_channel=mildlyoverfitted)
  
  - ***[jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)***:
    - A Pytorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

- ### Code notebook:

  - CNN:

    - [Kaggle: ResNet9 HMNIST Classification](https://www.kaggle.com/code/sidharthsinha/resnet9-hmnist-classification)
  - ViT:

    - [Kaggle: Vision Transformers in PyTorch | DeIT](https://www.kaggle.com/code/pdochannel/vision-transformers-in-pytorch-deit/notebook?scriptVersionId=85324242): This notebook trains a Vision Transformer on the Butterfly dataset.
    - [Kaggle: Swin Transformer in PyTorch](https://www.kaggle.com/code/pdochannel/swin-transformer-in-pytorch/notebook): This notebook trains a Vision Transformer on the Butterfly dataset.

## Params
- ### CNN
    - #### Resnet
    - #### vgg
    - #### [inception-V3](https://arxiv.org/abs/1512.00567)

- ### ViT 
    - #### DeIT: (deit_tiny_patch16_224.ipynb)
        - resize(224,224)
        - batch_size = 128

    - #### Swin-ViT: (swin_tiny_patch4_window7_224)
        - resize:(224,224)
        - batch_size = 64 
        
    - #### [CrossViT](https://github.com/rishikksh20/CrossViT-pytorch): 
        - size:224 (resize256 CenterCrop224)
        - batch_size = 64
