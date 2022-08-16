![deit_result1](/src/results/deit_result1.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = Cross_Entropy
- num_epochs = 10
- no data augmentation


![deit_result2](/src/results/deit_result2.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- no data augmentation

![deit_result3](/src/results/deit_result3.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation for each input data (ToPILImage, resize, CenterCrop, ToTensor, Normalize)

![deit_result4](/src/results/deit_result4.jpeg)
- optimizer = AdamW
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation
- size of train and test dataset = 7:3


10015 whole data set:
- num of each 'dx': 6705 1113 1099 514 327 142 115

train_dataset: 27306 - 34320


test_dataset: 24306 - 27305
- num of each 'dx': 2049 266 339 151 114 45 36 (comment的顺序 not code) nv....


