## Augmentation part:

chose geometric algorithms such as rotation(to a slight degree), clip, scale,
crop, resize as a realistic criteria for camera angle variation for a real world
scenarios.

This compromises the photometric algorithms(contrast, brightness, noise, guass blur).

augmentation should ensure a still humanly readable results and realistic variations
so all ranges of each function ensures no overkilling of the changed image

## Vectorization part:

We need Color Histogram(RGB or HSV) algorithm to capture color distribution info

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

after flattening was not enough(expected as it was not really descriptive),
we upgraded the base step by step
adding color histogram and HOG to increase descriptiveness of our generated vectors/features

after deciding on work flow for augmentation and feature extraction we
verified that we started getting better and better results with each tuning of our process

yet, reaching **0.85 accuracy** seemed to far from reach with our results that
varied around **0.65-0.75 accuracy** initially

## Parameter Tuning Journey:

trying to reach **0.85 accuracy**, we went through a very long
journey of trial and error, parameter tuning(for SVM, KNN, HOG, Color Histograms, Augmentation) but we needed to ensure reproducibility so we
identified many randomness along the way, augmentation is set with a random seed
and it shuffles the images of each raw folder data before augmenting them
as well as applying random number of augmentations per image that is varying
between **1 and 4** per image, choosing a random augmentation each time or per iteration, we cancelled our original approach of manually
splitting the train and test data and used train_test_split from sklearn
so we can set a random seed and startify the split to ensure each class is
represented equally, as well as chaning augmentation approach from simple **30%** each folder to an approach that is roughly equalizes the number of samples of each class to remove dominance, results are slowly improving although still far but
most importantly, they are reproducible thanks to setting random seeds along the way

after deciding on augmentation methodology, we thought about tuning feature extraction parameters
from image size tried **64, 80, 96, 128** to histogram params, we tried normal
BGR2RGB conversion, then moved to HSV which proved better, altered number of bins
per channel tried to give hue more importance, or balance them equally, tried
countless variations for HOG parameters, changing cell sizes, block sizes,
orientations, tried to give more importance to gradients by changing
the pixels_per_cell to smaller sizes, tried to give more importance to larger
structures by increasing the cells_per_block, even HOG normalization(L1, L2, L2-Hys) they give very different results with each configuration of everything, tried to change the
orientations to capture more directions, yet none of these changes
seemed to give a significant boost to our results, not even consistent improvements
everything varied depending on normalization steps, knn, svm configurations
so it was extremely hard and challenging to form a pattern or conclusion of what is actually
helping our models or what should we even use

## Final Tuning Session:

finally after the countless helpless trials, we decided to try one last
session of trail and error, it started with reviewing our augmentation

we decided to select which augmentation is realistic and helps our models

first is **noise** which we did not like at first as it degrades image quality
but after trying it, it actually helped our models to generalize better

then we added **blur**, which further helped generalization, then we figured
out that filling gaps after rotation is important then we noticed that
**flipping** is actually too bad for especially knn, removing it boosted results
and it was not that realistic anyway, finally keeping the set of augmentations
that proved helpful, kept randomness for reproducibility, we moved to feature extraction

decided to change image size once instead of per function then optimized between
different sizes, then decided good distribution of color histogram bins to capture
balance between color details and generalisation, finally HOG parameters
**8 pixels per cell, 2 cells per block** seemed to give best balanced
performance between knn and svm(svm actually reached **0.86** before that with small
vector size but knn was falling behind so we changed to this balanced approach)

finally after all these changes, we reached **0.85+ accuracy** 

## Random stars of our sucess:
**numpy normalization** after extracting hist, then after hog then again after concatenation, removing any degraded results.

using both **standard scaler and normalizer for svm** played a star role
which was not expected at all, alghough the same did not help knn

## Final works & notes:
**linear svm** was better before parameter tuning but proved much worse afterwards
**rbf svm** proved better after parameter tuning, so we kept it
then comes the rejection mechanism as our last challenge
for **knn**, since we decided on **manhattan distance**,
rejection logic is simple its like asking "are you in my neighborhood?"

if you are close to my k neighbors, then yes, otherwise you are an outsider.

so the code is getting distances to k neighbors, getting average distance
if average distance is larger than a threshold, then reject

again trail and error for what threshould to choose this way, knn with rejection
reached slightly worse than baseline **~0.846** when adding rejection.

for **svm**, analogy is even simpler, is the class outside the margin/border?

if you are far enough, then you are in my country, otherwise you are an outsider

we are using **rbf kernel** so its not flat border/margin but the logic is same

we are getting the distance from svm's decision function
then we are building a confidence score based on this distance

acceping, rejecting based on a threshold again trail and error

this way, svm reached actually better results with **0.877 accuracy** on accepted samples

noting this is not svm being better, rather it traded more rejections for better accuracy but overall models are very close in performance roughly similar.

acceping, rejecting based on a threshold again trail and error
this way, svm reached actually better results with 0.877 accuracy on accepted samples
noting this is not svm being better, rather it traded more rejections for better accuracy but overall models are very close in performance roughly similar.
rejection mechanism was the last challenge we faced in this project
as joblib and openCV make saving models, integrating to real world app
very straightforward, we are happy with our final results.

## Note:

Most of our detailed results are documented in the following files:

- `results/logs.txt` - (Almost)Complete run history(approaches are not always updated correctly)
- `results/accuracy-plots/` - Accuracy CSV files
- `results/classification-reports/` - Detailed classification metrics
- `results/confusion-matrices/` - Confusion matrix analysis
