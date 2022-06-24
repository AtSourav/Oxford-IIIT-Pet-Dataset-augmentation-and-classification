# Oxford-IIIT-Pet-Dataset-augmentation-and-classification

We take the Oxford-IIIT Pet Dataset, made available under a  [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), create an enlarged dataset with data augmentation, and then train some CNNs on this dataset.

### Dataset

The original dataset consists of 37 classes of pet images corresponding to different breeds of cats and dogs, with roughly 200 images per class. The annotations for the dataset give the species and breed id for each image, and they also provide trimap images that can be used to separate the foreground from the background. In addition to these a tight bounding box around the head of the animal is provided for each image but we do not make use of this in our project. 

### Data augmentation

Since there are 37 different classes and most of the classes are quite similar to each other (being different breeds of the same animal), about 150 images per class (saving some for the vaildation and the test set) are probably not enough to learn to distinguish between the classes well. So we'll generate some synthetic data using the images provided.

Alpha matting: Using the trimap images provided, we extract the foreground (pet) from a given image. This might help the network ignore the background in the classification process. 

Transformations: We generate some more images using random crop, random rotation, and random perspective.

As the images in the dataset are not of a uniform size (or aspect ratio), we resize the original and augmented images to a uniform size of (300 x 300). We have used a combination of 90 deg rotations and crop and resizing in the process. The rotated images are also good as ideally we want the network to treat the rotated image of a pet the same as a vertical one (although this probably makes the tast more difficult). For the transformations on the images, we have used the built in transforms in pytorch and skimage. We have written a simple function for the alpha matting using the trimaps and haven't used any existing packages.
