# The outline of the essay

## Introduction
Recent years, skin cancer has been one of the most common cancers worldwide. Skin cancer often develops from early pigmented lesions. This often requires a large number of dermatologists to pathologically diagnose whether the pigmented lesions are benign or cancerous. Therefore, if dermoscopy images can assist dermatologists to understand the imaging features of skin cancer and quickly gain experience in differential diagnosis, it will not only improve the efficiency and accuracy of diagnosis, but also save a lot of human resources. Dermoscopy images are also a suitable source for training artificial neural networks to automatically diagnose pigmented skin lesions. This paper will use Vision Transformer (ViT) related state-of-the-art structures compared with Convolutional Neural Networks (CNN) to classify skin disease types.

The skin cancer database used is HAM10000. 

CNN limitation: Convolutional neural networks (CNN), the most prevailing architecture for deep-learning based medical image analysis, are still functionally limited by their intrinsic inductive biases and inadequate receptive fields.




## Related Work
1. 2019: 
    - CNN: 
        - Indonesia researchers use CNN develop the identification system in 2019. The accuracy of training and testing reach 80% and 78% respectively.
2. 2020: 
    - CNN:
        - VGG19-based CNN and Transfer Learning(TL) proved to be powerful tools to aid skin cancer diagnosis with high accuracy.

3. 2021: 
    - CNN: 
        - Koreans: Apply fine-tuned Darknet and NasNet-mobile pre-trained model (Three datasets are used for the experimental process, such as **HAM10000**, ISBI2018, and ISBI2019 to achieve an accuracy of **95.8%**, 97.1%, and 85.35%, respectively.)
        - the accuracy of CNN model is around 75% for HAM10000 test set. 
    - ViT:
        - The attention models studied are completely modular and in this work they will be used with the popular ResNet architecture. The results show that attention modules do sometimes improve the performance of convolutional neural network architectures, but also that this improvement, although noticeable and statistically significant, is not consistent in different settings. (Accuracy around 70% the highest ViT-Base reach 0.737 with params 85.80M)

4. 2022: 
    - CNN: 
        - Taiwan researchers use InceptionResNetV2 model provided by Google to make a differential diagnosis (no cited)
        - In the classification tasks, VGG16 and MobileNet V2 CNN models were fine-tuned and trained through TL on dermoscopic images. (Koreans)
        - the proposed CNN has obtained an accuracy of 95.18%
    
    - ViT:
        - multimodal transformer
        - swin transformer
        - [DeiT](https://paperswithcode.com/paper/deit-iii-revenge-of-the-vit)
        - [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://paperswithcode.com/paper/2103-14899) For example, on the ImageNet1K dataset, with some architectural changes, our approach outperforms the recent DeiT

5. SOTA architecture to do image classification:
    - [Sequencer: Deep LSTM for Image Classification](https://paperswithcode.com/paper/sequencer-deep-lstm-for-image-classification)


## Method
- What is [Vision transformer](https://paperswithcode.com/method/vision-transformer)?

    The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

- What is [DeiT](https://paperswithcode.com/paper/deit-iii-revenge-of-the-vit)


- How to do the data preprocessing?

    - how to do data augmentation?






## Experiment
- CNN vs ViT

- ViT Analysis:
    - experiment results (metric): Accuracy, Precision, Recall, F1-score

- Fined Tuned ViT Analysis:
    - Compare different loss function
    - Compare different optimizer (torch.optim.SGD vs )
    - Compare different regularization (dropout vs L2 vs L1)
     
     


## Conclusion 
- Future Work:
    - Preprocessing: remove hair to enhance the training accuracy further
    
