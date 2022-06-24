# Oxford-IIIT-Pet-Dataset-augmentation-and-classification

We take the Oxford-IIIT Pet Dataset, made available under a  [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), create an enlarged dataset with data augmentation, and then train some CNNs on this dataset using pytorch. 

(include the best results, mention svm results for comparison)

### Dataset

The original dataset consists of 37 classes of pet images corresponding to different breeds of cats and dogs, with roughly 200 images per class. The annotations for the dataset give the species and breed id for each image, and they also provide trimap images that can be used to separate the foreground from the background. In addition to these a tight bounding box around the head of the animal is provided for each image but we do not make use of this in our project. 

### Data augmentation

Since there are 37 different classes and most of the classes are quite similar to each other (being different breeds of the same animal), about 150 images per class (saving some for the vaildation and the test set) are probably not enough to learn to distinguish between the classes well. So we'll generate some synthetic data using the images provided.

Alpha matting: Using the trimap images provided, we extract the foreground (pet) from a given image. This might help the network ignore the background in the classification process. 

Transformations: We generate some more images using random crop, random rotation, and random perspective.

As the images in the dataset are not of a uniform size (or aspect ratio), we resize the original and augmented images to a uniform size of (300 x 300). We have used a combination of 90 deg rotations and crop and resizing in the process. The rotated images are also good as ideally we want the network to treat the rotated image of a pet the same as a vertical one (although this probably makes the tast more difficult). For the transformations on the images, we have used the built in transforms in pytorch and skimage. We have written a simple function for the alpha matting using the trimaps and haven't used any existing packages. Finally we save all the processed images externally. data_augment.ipynb is the relevant notebook.

### Datasets

Using all the original and augmented sets of images we create different datasets for training (and validation, testing), so we can compare the performance of the model trained on the different datasets too. The datasets are different in total size, train:validation:test ratios, and also in the proportion of augmented images used. The validation and the test sets are always composed entirely of the original images (cropped and resized). We save the annotations for these different datasets externally in the form of a csv file. These annotation files are contained in the folder annotations_aug/. There is a csv file for the train, valid, and test set in each case (evident from the file names). Let's summarize the different datasets below, the details are present in the notebook datasets.ipynb (not necessarily in the order they're presented here):

1. **Set O**: This is composed entirely of the original images. There are 7390 images in total, and the train:valid:test ratio is roughly like 85:10:5. There are around 180 images per class in the training set. The annotation files are named annotations_train_O.csv, annotations_valid_O.csv, and annotations_test_O.csv respectively.

2. **Set r1**: This is composed of the original and alpha matted images only, roughly in the ratio 3:2. There are approximately 190 images per class in the training set, and the train:valid:test ratio is roughly like 70:15:15. The annotations file for the training set is named annotations_train_r1.csv. The validation and test set (which are the same for the sets r2 and r3 listed below) annotation files are named annotations_valid.csv and annotations_test.csv respectively.

3. **Set r2**: It's just like the set r1 with the ratio of original to alpha matted images being around 1:1.

4. **Set r2**: Similar to sets r1 and r2, the ratio of original to alpha matted images is now 2:3 approximately.

5. **Set L**:  : This set is also composed from the original and alpha matted images only but has a larger training set. The train:valid:test ratio is now around 89:7:4 with roughly 320 images per class in the training set. The annotations for the training set are contained in annotations_train_Lall.csv, while the validation and test set annotation files are named annotations_valid_L.csv and annotations_test_L.csv respectively.

6. **Set B1**: This is composed of the original, alpha matted, and the otherwise transformed new images in equal proportions with roughly 450 images per class in the training set. The train:valid:test ratio is approximately 90:6:4. The training set annotations file is called annotations_train_B1.csv and the corresponding files for the validation and test sets are named accordingly.

7. **Set B2**: This is just like the set B1 but with a larger training set with about 480 images per class in it. The train:valid:test ratio is around 92:5:3.

### Experiments

We mostly tried CNNs with 4-7 convolutional layers and one dense layer at the end (with batch normalization, dropout, and L2 regularization). In a few cases we tried to use two dense layers but this didn't improve the results. And we prefer to stick to convolutional layers as they are lighter than dense layers. The most successful model in all cases was the one with 6 conv layers and one dense layer. We used a minibatch size of 64, and used the Adam optimizer on the NLL loss function.

Overall, the validation set performance (in terms of loss and accuracy) always seems to saturate at some value and remains mostly constant with further training. The saturation values seems to improve upon choosing a larger training set. If trained for long enough (and unless the regularization is too strong), the model can always overfit the training data. We don't present the learning curve for each and every experiment here, but they are similar to the ones in the notebooks trainB1.ipynb, trainB2.ipynb, and trainL.ipynb, with different saturation values that we summarize below. The initialization of weights was random, and different runs on the same model sometimes came up with slightly different results.

Increasing the regularization did not in general improve the saturation value for the accuracy on the validation set. Mostly it slowed the learning process on the training set (and in some cases caused the training set accuracy to saturate as well), but it didn't really improve performance on the validation set. This seems to suggest that the problem is that the training set doesn't represent the validation set very well, and we need a better and larger dataset to obtain better results on this problem.

Let's summarize the different experiments and the results here:

1. **Set r1**: Trained CNNs with 4-7 conv layers, with best results obtained with 6 conv layers and one dense layer (almost exactly as in the notebook trainL.ipynb) with a max validation accuracy of ~40%. We also tried with two or three dense layers but it didn't help improve results.

2. **Set r2**: Trained on a model with 6 conv layers and one dense layer (suggested by results from set r1), and obtained a max validation accuracy of ~37%.

3. **Set r3**: Similar results as r2. We can conclude that for a training set of this size, the proportion of original images to alpha matted images did not make a difference significant enough for us to be able to draw any conclusions.

4. **Set O**: The 6 conv layer network had a max performance of ~35% on the validation set.

5. **Set L**: Obtained a max validation accuracy of ~47% using the 6 conv 1 dense layer network.

6. **Set B1**: Obtained a max validation accuracy of ~54% using the 6 conv 1 dense layer network. Needed to use weaker regularization than the ones above for the model to overfit on the training set (probably coz the training set is noisy already from the transforms used in the augmented images). Also used learning rate decay to smoothen the learning curves that were highly oscillatory otherwise. Also tried using a different number of conv layers as mentioned above but performance was always worse. 

7. **Set B2**: This is the largest training set of all, and the observations are similar to the ones made with the set B1. The max validation accuracy obtained (with the same model as above) was ~57%.


