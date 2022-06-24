# Oxford-IIIT-Pet-Dataset-augmentation-and-classification

We take the Oxford-IIIT Pet Dataset, made available under a  [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), create an enlarged dataset with data augmentation, and then train some CNNs on this dataset. 

(include the best results)

### Dataset

The original dataset consists of 37 classes of pet images corresponding to different breeds of cats and dogs, with roughly 200 images per class. The annotations for the dataset give the species and breed id for each image, and they also provide trimap images that can be used to separate the foreground from the background. In addition to these a tight bounding box around the head of the animal is provided for each image but we do not make use of this in our project. 

### Data augmentation

Since there are 37 different classes and most of the classes are quite similar to each other (being different breeds of the same animal), about 150 images per class (saving some for the vaildation and the test set) are probably not enough to learn to distinguish between the classes well. So we'll generate some synthetic data using the images provided.

Alpha matting: Using the trimap images provided, we extract the foreground (pet) from a given image. This might help the network ignore the background in the classification process. 

Transformations: We generate some more images using random crop, random rotation, and random perspective.

As the images in the dataset are not of a uniform size (or aspect ratio), we resize the original and augmented images to a uniform size of (300 x 300). We have used a combination of 90 deg rotations and crop and resizing in the process. The rotated images are also good as ideally we want the network to treat the rotated image of a pet the same as a vertical one (although this probably makes the tast more difficult). For the transformations on the images, we have used the built in transforms in pytorch and skimage. We have written a simple function for the alpha matting using the trimaps and haven't used any existing packages. Finally we save all the processed images externally. data_augment.ipynb is the relevant notebook.

### Datasets

Using all the original and augmented sets of images we create different datasets for training (and validation, testing), so we can compare the performance of the model trained on the different datasets too. The datasets are different in total size, train:validation:test ratios, and also in the proportion of augmented images used. The validation and the test sets are always composed entirely of the original images (cropped and resized). We save the annotations for these different datasets externally in the form of a csv file. These annotation files are contained in the folder annotations_aug/. There is a csv file for the train, valid, and test set in each case (evident from the file names). Let's summarize the different datasets below, the details are present in the notebook datasets.ipynb (not necessarily in the order they're presented here):

1. annotations_train_O.csv : This is composed entirely of the original images. There are 7390 images in total, and the train:valid:test ratio is roughly like 85:10:5. There are around 180 images per class in the training set. The corresponding validation and test set files are named annotations_valid_O.csv and annotations_test_O.csv respectively.

2. annotations_train_r1.csv : This is composed of the original and alpha matted images only, roughly in the ratio 3:2. There are approximately 190 images per class in the training set, and the train:valid:test ratio is roughly like 70:15:15. The validation and test set (which are the same for the sets r2 and r3 listed below) annotation files are named annotations_valid.csv and annotations_test.csv respectively.

3. annotations_train_r2.csv : It's just like the set r1 with the ratio of original to alpha matted images being around 1:1.

4. annotations_train_r3.csv : Similar to sets r1 and r2, the ratio of original to alpha matted images is now 2:3 approximately.

5. annotations_train_Lall.csv : This set is also composed from the original and alpha matted images only but has a larger training set. The train:valid:test ratio is now around 89:7:4 with roughly 320 images per class in the training set. The validation and test set annotation files are named annotations_valid_L.csv and annotations_test_L.csv respectively.

6.   
