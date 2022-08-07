# Study Notes
## Small points
- ### DataLoader(Preprocessing):
  - [第五章——Pytorch中常用的工具](https://blog.csdn.net/zhenaoxi1077/article/details/80953227)：torch.utils.data.Dataset, torchvision.transforms, torch.utils.data.DataLoader, torch.utils.data.sampler

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

