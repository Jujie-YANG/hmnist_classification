# Study Notes
## Small points
- ### DataLoader(Preprocessing):
  - [第五章——Pytorch中常用的工具](https://blog.csdn.net/zhenaoxi1077/article/details/80953227)：torch.utils.data.Dataset, torchvision.transforms, torch.utils.data.DataLoader, torch.utils.data.sampler

  - [```torchvision.transforms.RandomApply(transforms, p=0.5)```](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomApply.html#torchvision.transforms.RandomApply): Apply randomly a list of transformations with a given probability
  [```torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)```](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html): Randomly change the brightness, contrast, saturation and hue of an image.
  [```torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)```]: Randomly selects a rectangle region in an torch Tensor image and erases its pixels.

- ### Define Model:


- ### Training:

    - Optimizer (Update params):
        - [Understand PyTorch optimizer.param_groups with Examples – PyTorch Tutorial](https://www.tutorialexample.com/understand-pytorch-optimizer-param_groups-with-examples-pytorch-tutorial/): optimizer.param_groups is a python list, which contains a dictionary

        - [Kaggle explain: One-cycle learning rate schedulers](https://www.kaggle.com/code/residentmario/one-cycle-learning-rate-schedulers/notebook): This cyclic learning rate policy is meant to be applied over one entire learning cycle: **e.g. one epoch**. Fast.AI calls this the one cycle training. After each cycle, you are supposed to re-apply the learning rate finder to find new good values, and then fit another cycle, until no more training occurs; hence the name. Use ```optimizer.step()``` before ```scheduler.step()```. Also, for ```OneCycleLR```, you need to run ```scheduler.step()``` after every step.

        - [```optimizer.step()```](https://pytorch.org/docs/stable/optim.html#optimizer-step): Below is a simplified version supported by most optimizers. The function can be called once the gradients are computed using e.g. backward()
            ```
            for input, target in dataset:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            ```

    - Prevent overfitting / avoid exploding gradient (Optimizer):
        - [Regularization in Deep Learning — L1, L2, and Dropout](https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036#:~:text=Regularization%20is%20a%20set%20of,data%20from%20the%20problem%20domain.): Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the accuracy of a Deep Learning model when facing completely new data from the problem domain. Normalisation adjusts the data; regularisation adjusts the prediction function.
        - [Deep learning basics — weight decay](https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9): ```optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)``` Aim: 1. To prevent overfitting 2. To keep the weights small and avoid exploding gradient
    
        - [Understanding Gradient Clipping (and How It Can Fix Exploding Gradients Problem)](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem): Gradient clipping-by-value + Gradient clipping-by-norm e.g. ```nn.utils.clip_grad_value_(model.parameters(), grad_clip)```

- ### Inference/Validation:

  - [YouTube Video: 14- Pytorch: What is model.eval?](https://www.youtube.com/watch?v=GzjRE3MUx6Q) - model.eval() disables dropout and batch normalization, the gradients are still computed as model.train(). model.eval() is necessary for inference.
  - [Use of ‘model.eval()’ and ‘with torch.no_grad()’ in PyTorch model evaluate](https://androidkt.com/use-of-model-eval-and-with-torch-no_grad-in-pytorch-model-evaluate/): Dropout Layer, Batch Normalization Layer(nn.BatchNorm2d()), with torch.no_grad()

    - model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    - torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).


## Notes
- ```plt.imshow(np.transpose(img.numpy(),(1,2,0)))``` : plt: (width, height, channel) e.g. torch.size(3,64,64) -> (64,64,3)
- [Difference between view, reshape, transpose and permute in PyTorch](https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/): 
        ```
        x = torch.rand(16, 32, 3)
        y = x.tranpose(0, 2)
        z = x.permute(2, 1, 0)
        ```
- [When to use yield instead of return in Python?](https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/): **Return** sends a specified value back to its caller whereas **Yield** can produce a sequence of values. We should use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory. Yield is used in Python **generators**.

- [torch.squeeze and torch.unsqueeze – usage and code examples](https://linuxpip.org/pytorch-squeeze-unsqueeze/): The **squeeze** method "returns a tensor with all the dimensions of input of size 1 removed", while **unsqueeze** "returns a new tensor with a dimension of size one inserted at the specified position"

## Metrics (F1 Score)
- [Confusion Matrix, Accuracy, Precision, Recall, F1 Score](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd)

- [F-1 Score for Multi-Class Classification](https://www.baeldung.com/cs/multi-class-f1-score)

- [F1 Score vs ROC AUC vs Accuracy vs PR AUC: Which Evaluation Metric Should You Choose?](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)

- [Accuracy vs. F1-Score](https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2): **1.** Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial **2.** Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes as in the above case. **3.** In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on.

- [sklearn.metrics.precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html): Compute precision, recall, F-measure and support for each class.

## Learning Materials about Transformer
### - Tutorials
- [Bilibili: 唐宇迪带你从零详细解读Transformer模型](https://www.bilibili.com/video/BV1Pu411Q7jD?spm_id_from=333.999.0.0&vd_source=4e20016bd1355fe9ad9e32194a97d42a)

- [YouTube: BERT Neural Network - EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo)

- [YouTube: Transformer Model (1/2): Attention Layers](https://www.youtube.com/watch?v=FC8PziPmxnQ)
- [YouTube: Transformer Model (2/2): Build a Deep Neural Network](https://www.youtube.com/watch?v=J4H6A4-dvhE)

### - Blogs
- [Transformers Explained Visually (Part 1): Overview of Functionality](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

- [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34)

- [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

- [Transformers Explained Visually — Not Just How, but Why They Work So Well](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3)

- [CSDN:【解析】Vision Transformer 在图像分类中的应用](https://blog.csdn.net/ViatorSun/article/details/115586005)

## Learning Materials about CNN
### - Websites
- [PyTorch official website: Train a CNN classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and its corresponding [colab code notebook](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/17a7c7cb80916fcdf921097825a0f562/cifar10_tutorial.ipynb?hl=en#scrollTo=Mwf3hCS855qn)


## Colab Use
- [How to save our model to Google Drive and reuse it](https://medium.com/@ml_kid/how-to-save-our-model-to-google-drive-and-reuse-it-2c1028058cb2):
```
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# Save
torch.save(model.state_dict(), '/content/drive/MyDrive/kaggle/save_model/deit_tiny_patch16_224.pth')
# Load
model.load_state_dict(torch.load(path))
```

## Others
- [GitHub repo: pytorch/examples](https://github.com/pytorch/examples):  is a repository showcasing examples of using PyTorch in Vision, Text, Reinforcement Learning, etc.

