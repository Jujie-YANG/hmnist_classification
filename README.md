# hmnist_classification

## Dataset

[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): a large collection of multi-source dermatoscopic images of pigmented lesions (Downloaded in the Kaggle)

## Useful Links:


### ViT related materials:
- Transformer tutorials:
    - [Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
    - [Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

- ViT Documentation:
    - [timm Pytorch Documentation](https://rwightman.github.io/pytorch-image-models/)

- ViT example code notebook:
    [Kaggle: Vision Transformers in PyTorch | DeIT](https://www.kaggle.com/code/pdochannel/vision-transformers-in-pytorch-deit/notebook?scriptVersionId=85324242)
    

### Small points

- Training:

  - [第五章——Pytorch中常用的工具](https://blog.csdn.net/zhenaoxi1077/article/details/80953227)：torch.utils.data.Dataset, torchvision.transforms, torch.utils.data.DataLoader, torch.utils.data.sampler
  - [Regularization in Deep Learning — L1, L2, and Dropout](https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036#:~:text=Regularization%20is%20a%20set%20of,data%20from%20the%20problem%20domain.): Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the accuracy of a Deep Learning model when facing completely new data from the problem domain. Normalisation adjusts the data; regularisation adjusts the prediction function.
  - [Deep learning basics — weight decay](https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9): ```optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)```
  - [Understanding Gradient Clipping (and How It Can Fix Exploding Gradients Problem)](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem): Gradient clipping-by-value + Gradient clipping-by-norm
- Inference/Validation:

  - [YouTube Video: 14- Pytorch: What is model.eval?](https://www.youtube.com/watch?v=GzjRE3MUx6Q) - model.eval() disables dropout and batch normalization, the gradients are still computed as model.train(). model.eval() is necessary for inference.
  - [Use of ‘model.eval()’ and ‘with torch.no_grad()’ in PyTorch model evaluate](https://androidkt.com/use-of-model-eval-and-with-torch-no_grad-in-pytorch-model-evaluate/): Dropout Layer, Batch Normalization Layer(nn.BatchNorm2d()), with torch.no_grad()

    - model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    - torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).

## Notes
- ```plt.imshow(np.transpose(img.numpy(),(1,2,0)))``` : plt: (width, height, channel) e.g. torch.size(3,64,64) -> (64,64,3)
- [When to use yield instead of return in Python?](https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/): **Return** sends a specified value back to its caller whereas **Yield** can produce a sequence of values. We should use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory. Yield is used in Python **generators**.
