![deit_result1](/src/results/deit_result1.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = Cross_Entropy
- num_epochs = 10
- no data augmentation


![deit_result1](/src/results/deit_result2.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- no data augmentation

![deit_result1](/src/results/deit_result3.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation for each input data (ToPILImage, resize, CenterCrop, ToTensor, Normalize)