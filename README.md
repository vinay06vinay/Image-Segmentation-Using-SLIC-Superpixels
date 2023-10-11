# Image-Segmentation-Using-SLIC-Superpixels

## Introduction:
The primary goal of the project is to first implement the SLIC algorithm to segment the image using Superpixels. SLIC is a superpixel segmentation technique that groups pixels in an image into visually meaningful regions called superpixels. These superpixels are more perceptually relevant than regular grid-based pixel subdivisions, making them useful in various computer vision and image analysis tasks.SLIC works by clustering pixels based on their color similarity and proximity in the image, resulting in a compact and efficient representation of the image that simplifies subsequent processing tasks. 

The secondary goal is to build a segmentation network, which uses SLIC Superpixels as input. In essense, it will be a classifier for superpixels. The end product is a system which, when given an image, computes superpixels using SLIC and classifies each superpixel as one of the 10 classes of MSRC v1. This task is achieved through using pre-trained VGG16 by replacing the last few layers of the network with fully connected layers to classify the 10 classes.

## Implementation of Segmentation Network:
1. Firstly, after building the SLIC algorithm in reference to "https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf".
2. For each image,
    - Get superpixels sp_i for image x. We adopt 100 segments in this assignment, 'segments = slic(image, n_segments=100, compactness=10)'.
    For every superpixel sp_i in the image,
      - find the smallest rectangle which can enclose sp_i
      - Dilate the rectangle by 3 pixels.
      - Get the same region from the segmentation image (from the file with similar name with *_GT). The class for this sp_i is mode of segmentation classes in that same region. Save the              dilated region as npy (jpg is lossy for such small patches).
3. Once the Superpixels patch for each image are labeled according to the ground truth, these superpixels are sent through a pre-trained VGG16 network to be able to classify each patch on the 10 classes in the MSRC v1 dataset.

### Code:
The code for the whole implementation can be found in "SLIC-Segmentation-Network.ipynb"
