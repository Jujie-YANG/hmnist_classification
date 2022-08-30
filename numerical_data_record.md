<!-- ![deit_result1](/src/results/deit_result1.jpeg)
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
- size of train and test dataset = 7:3 -->

## Index of image number for three experimental groups:
- size: 7:3
    - train_dataset: 27306 - 34320
    - test_dataset: 24306 - 27305
- size: 8:2
    - train_dataset: 26306 - 34320
    - test_dataset: 24306 - 26305
- size: 8:2
    - train_dataset: 25306 - 34320
    - test_dataset: 24306 - 25305

## num of each 'dx'(nv, mel, bkl, bcc, akiec, vasc and df) comment从大到小的顺序，不是code里0-6 classes的顺序:
- 10015 whole dataset: 6705 1113 1099 514 327 142 115
- train_dataset size per class:
    - 7:3: 4656 847 760 363 213 97 79
    - 8:2: 5334 941 870 416 252 110 92
    - 9:1: 6020 1022 988 462 291 128 104
- test_dataset size per class:
    - 7:3: 2049 266 339 151 114 45 36
- code里0-6 classes的顺序:
    - 0 == "akiec"
    - 1 == "bcc"
    - 2 == "bkl"
    - 3 == "df"
    - 4 == "mel"
    - 5 == "nv"
    - 6 == "vasc"

## Max Acc for all models (Three ViT models use the same params as ResNet9):
- 7:3:
    - Resnet9: 78.28
    - Deit: 75.19
    - Swin: 69.25
    - CrossViT: 76.07
- 8:2：
    - Resnet9: 78.12
    - Deit: 77.34
    - Swin: 72.59
    - CrossViT: 76.57
- 9:1:
    - Resnet9: 78.82
    - Deit: 75.72
    - Swin: 68.61
    - CrossViT: 78.78

## Modified params for three ViT models(corresponds to Official Website) -- All no gradient clip
- DeiT:
    - optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=**1e-8**, weight_decay=**0.05**)
    - sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=**1e-5**)
    - criterion = LabelSmoothingCrossEntropy()

- Swin:
    - optimizer = torch.optim.AdamW(model.parameters(), lr=**5e-4**, weight_decay=**0.05**)
    - sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=**5e-6**, last_epoch=-1, verbose=False)
    - criterion = LabelSmoothingCrossEntropy(**smoothing=0.1**)

- CrossViT:
    - optimizer = torch.optim.AdamW(model.parameters(), lr=**5e-4**, eps=**1e-8**, weight_decay=0.05)
    - sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=**5e-6**, last_epoch=-1, verbose=False)
    - criterion = LabelSmoothingCrossEntropy()


- ### 7:3
    <!-- - DeiT: 78.85 (lr=0.001) -->
    - DeiT: 82.43 (lr=5e-4)
    - Swin: 83.72
    - CrossViT: 76.62

- ### 8:2
    <!-- - DeiT: 80.48 (lr=0.001) -->
    - DeiT: 81.12(lr=5e-4)
    - Swin: 84.35
    - CrossViT: 77.86

- ### 9:1:
    <!-- - DeiT: 80.47 (lr=0.001) -->
    - DeiT: 85.07 (lr=5e-4)
    - Swin: 84.92
    - CrossViT: 79.02

## A table shows the results of overall accuracy and F1-score per class
| Models            | Overall Accuracy | f_score for akiec | f_score for bcc | f_score for bkl| f_score for df | f_score for mel | f_score for nv| f_score for vasc|
| :----:  |   :----:  |    :----:   |    :----:  |  :----:  |   :----:  |   :----:   |  :----:  |   :----:  |
| Resnet9           |             |               | | | | | | |
| Deit              |             |               | | | | | | |
| Swin Transformer  |             |               | | | | | | |
| CrossViT          |             |               | | | | | | |
 
