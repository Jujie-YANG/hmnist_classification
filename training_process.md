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

![deit_result5](/src/results/deit_result5.jpeg)
- optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001) change to model.parameters()
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation (resize 256, centerCrop 224)
- size of train and test dataset = 7:3

![deit_result6](/src/results/deit_result6.jpeg)
- optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001) change to model.parameters()
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation (resize 224,224 directly, without centerCrop)
- size of train and test dataset = 7:3

![resnet9_result1](/src/results/resnet9_result1.jpeg)
- optimizer = torch.optim.Adam
- scheduler = OneCycleLR
- criterion = CrossEntropy
- data augmentation (resize 32,32)
- size of train and test dataset = 7:3

![swin_result1.jpeg](/src/results/swin_result1.jpeg)
- model name = swin_tiny_patch4_window7_224
- optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation (resize 256, centerCrop 224)
- size of train and test dataset = 7:3

![crossvit_result1.jpeg](/src/results/crossvit_result1.jpeg)
- model name = swinv2_tiny_patch4_window8_256
- optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
- scheduler = StepLR
- criterion = LabelSmoothingCrossEntropy
- num_epochs = 10
- data augmentation (resize 256, centerCrop 224)
- size of train and test dataset = 7:3

10015 whole dataset:
- num of each 'dx': 6705 1113 1099 514 327 142 115

train_dataset: 27306 - 34320
- num of each 'dx': 4656 847 760 363 213 97 79 (not calculated by program)

test_dataset: 24306 - 27305
- num of each 'dx': 2049 266 339 151 114 45 36 (comment的顺序 not code: 'nv',....) - calculated by excel function


