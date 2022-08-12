# hmnist_classification

## Dataset
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): a large collection of multi-source dermatoscopic images of pigmented lesions (Downloaded in the Kaggle)

## ViT related materials:
[Vision Transformer(paperswithcode)-- refer to this website for more SOTA models](https://paperswithcode.com/method/vision-transformer): The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used. 

- ### Transformer tutorials:
    - [Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
    - [Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

- ### ViT Documentation:
    - [timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)

- ### GitHub models:


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