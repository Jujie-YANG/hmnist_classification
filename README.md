# hmnist_classification

## Dataset
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), a large collection of multi-source dermatoscopic images of pigmented lesions


## Useful Links:
### Awesome Tutorials about Transformer and ViT
- [Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
- [Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

- [timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)


### Small points
- [14- Pytorch: What is model.eval?](https://www.youtube.com/watch?v=GzjRE3MUx6Q) - model.eval() disables dropout and batch normalization, the gradients are still computed as model.train(). model.eval() is necessary for inference.

- [Pytorch: torch.utils.data.Dataset, torchvision.transforms, torch.utils.data.DataLoader, torch.utils.data.sampler](https://blog.csdn.net/zhenaoxi1077/article/details/80953227)

plt.imshow(np.transpose(img.numpy(),(1,2,0))) -- plt: (width, height, channel)
torch.size(3,64,64) -> (64,64,3)