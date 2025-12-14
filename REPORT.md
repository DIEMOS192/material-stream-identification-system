Augmentation part:

chose geometric algorithms such as rotation(to a slight degree), clip, scale,
crop, resize as a realistic criteria for camera angle variation for a real world
scenarios.
This compromises the photometric algorithms.
augmentation should ensure a still humanly readable results and realistic variations
any non-realistic augmentation shall be immediately deleted from the training set.

Vectorization part:

We need Color Histogram(RGB) algorithm to capture color distribution info
We need HOG(Gray Scale) to capture edges/textures, should be robust against
small rotations or flips

Concatenate the 2 to generate feature vector per image

this approach should be sufficiently descriptive and descrimintive for
categories and features of the images

use RBS for SVM Kernel (verify that its actually better)

use small K for KNN (again verify that small k is actually better)

each class should have similar number of examples to reach good accuracy

we will initiate with a basic numpy flattening(including resizing, normalization) to have a baseline on feature extraction
before moving to more advanced approaches such as HOG



flagged*We initially used raw pixel intensities with normalization as a baseline feature representation to establish a stable reference performance before exploring more descriptive features.*flagged