# Oxford-IIIT-Pet-Dataset-augmentation-and-classification

We take the Oxford-IIIT Pet Dataset, made available under a  [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), create an enlarged dataset with data augmentation, and then train some CNNs on this dataset.

### Dataset

The original dataset consists of 37 classes of pet images corresponding to different breeds of cats and dogs, with roughly 200 images per class. The annotations for the dataset give the species and breed id for each image, and they also provide trimap images that can be used to separate the foreground from the background. In addition to these a tight bounding box around the head of the animal is provided for each image but we do not make use of this in our project. 
