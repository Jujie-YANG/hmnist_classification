# hmnist_classification

## Dataset
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): a large collection of multi-source dermatoscopic images of pigmented lesions (Downloaded in the Kaggle)

## ViT related materials:
[Vision Transformer(paperswithcode)-- refer to this website for more SOTA models](https://paperswithcode.com/method/vision-transformer): The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used. 

- ### Transformer tutorials:
    - [Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
    - [Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

- ### ViT Documentation:
    - ***[timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)***: Py**T**orch **Im**age **M**odels (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results. -- All model architecture families include variants with pretrained weights.

        [pytorch-image-models/timm/models/vision_transformer.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

        Reference code and pretrained weights:[Google's repository for the ViT model](https://github.com/google-research/vision_transformer)

- ### GitHub models:
    - ***[timm implementation](https://github.com/rwightman/pytorch-image-models)***: 
    
        About: PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more refer to the [timm documentation](https://rwightman.github.io/pytorch-image-models/)

    - ***[lucidrains implementation](https://github.com/lucidrains/vit-pytorch)***: 
        
        About: Implementation of Vision Transformer, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Pytorch.

        Significance is further explained in [Yannic Kilcher's](https://www.youtube.com/watch?v=TrdevFK_am4) video.

        For a Pytorch implementation with pretrained models, please see Ross Wightman's repository [here(timm implementation)](https://github.com/rwightman/pytorch-image-models)

        The official Jax repository is [here(Google's repository for the ViT model)](https://github.com/google-research/vision_transformer).

    - ***[jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)***:
        
        Pytorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

        This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

- ### Code notebook:
    - CNN:
        - [Kaggle: ResNet9 HMNIST Classification](https://www.kaggle.com/code/sidharthsinha/resnet9-hmnist-classification)

    - ViT:
        - [Kaggle: Vision Transformers in PyTorch | DeIT](https://www.kaggle.com/code/pdochannel/vision-transformers-in-pytorch-deit/notebook?scriptVersionId=85324242): This notebook trains a Vision Transformer on the Butterfly dataset.

        - [Kaggle: Swin Transformer in PyTorch](https://www.kaggle.com/code/pdochannel/swin-transformer-in-pytorch/notebook): This notebook trains a Vision Transformer on the Butterfly dataset.
    


## Params

### - DeIT: (deit_tiny_patch16_224.ipynb)
- resize(224,224) 
- batch_size = 128

### - Swin-ViT: ()
- resize:(224,224)
- batch_size = 64 (RuntimeError: CUDA out of memory. Tried to allocate 74.00 MiB (GPU 0; 14.76 GiB total capacity; 13.40 GiB already allocated; 67.75 MiB free; 13.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF)