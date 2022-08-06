# hmnist_classification

## Dataset
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), a large collection of multi-source dermatoscopic images of pigmented lesions


## Useful Links:
### Awesome Tutorials about Transformer and ViT
- [Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
- [Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

- [timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)


### Small points
- [YouTube Video: 14- Pytorch: What is model.eval?](https://www.youtube.com/watch?v=GzjRE3MUx6Q) - model.eval() disables dropout and batch normalization, the gradients are still computed as model.train(). model.eval() is necessary for inference.

- [第五章——Pytorch中常用的工具](https://blog.csdn.net/zhenaoxi1077/article/details/80953227)：torch.utils.data.Dataset, torchvision.transforms, torch.utils.data.DataLoader, torch.utils.data.sampler

- [Regularization in Deep Learning — L1, L2, and Dropout](https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036#:~:text=Regularization%20is%20a%20set%20of,data%20from%20the%20problem%20domain.): Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the accuracy of a Deep Learning model when facing completely new data from the problem domain.

- [Use of ‘model.eval()’ and ‘with torch.no_grad()’ in PyTorch model evaluate](https://androidkt.com/use-of-model-eval-and-with-torch-no_grad-in-pytorch-model-evaluate/): Dropout Layer, Batch Normalization Layer(nn.BatchNorm2d())


## Notes
- ```plt.imshow(np.transpose(img.numpy(),(1,2,0)))``` -- plt: (width, height, channel) e.g. torch.size(3,64,64) -> (64,64,3)