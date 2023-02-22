# Barcode Detection Application
## Description
 It is a barcode detection algorithm that detects all possible objects and the barcode printed on them. It is made possible through the use of the OpenCV library.
 
## Algorithm
The main idea is contour detection.
* First the high-resolution image is converted to grayscale. The copy of the image is stored for later post-processing.
* Adaptive Thresholding is used to counter the change in brightness and the conversion into a binary image.
* The obtained binary image is inverted to detect the counters present in the image and the bounding box around them is predicted.
* The boxes with reasonably small areas are filtered out before predicting any barcodes in them.
* To avoid loss through resizing, a list of cropped images is passed to detect any barcode within those.
* Finally, while post-processing non-maximal suppression is used to remove the overlapping bounding boxes, and the bounding boxes are drawn with suitable color code logic.
```
Black- for all the objects present in an image.
Red - for items that did not have or couldn't detect any barcode
Blue- for items whose barcode was read successfully.
```

## Results-
Bounding boxes and the output for each image
![](https://i.imgur.com/8HkDJiE.jpg)
```
The detected barcodes and their unique count
{b'8901030574252': 2, b'8901030656026': 1, b'8901719112737': 1}
```

![](https://i.imgur.com/iScmJMr.jpg)
```
The detected barcodes and their unique count
{b'8901030578199': 1,
 b'8901030656026': 1,
 b'8901063160088': 3,
 b'8901725114916': 2}
```

![](https://i.imgur.com/SZoz5Ho.png)
```
The detected barcodes and their unique count
{b'8901719110856': 2}
```

![](https://i.imgur.com/U9D48QW.png)
```
The detected barcodes and their unique count
{}
```
![](https://i.imgur.com/LTv5hCR.png)
```
The detected barcodes and their unique count
{}
```

## Installation-
1. Clone the repository
2. Navigate to the cloned repository(Note: For Linux,before installing Pyzbar first run this command to avoid import error - `apt install libzbar0`)
3. Run command `$ pip install -e ./`
The following dependencies could also be installed directly.
```
OpenCV
Matplotlib
NumPy
Pyzbar
```
4. Simply, change the path of the input and final output image in `contour.py` and run it.

## Limitations-
1. Since this approach doesn't use any pre-trained model, classification between different items is not possible.
2. This approach considers the partial detection of barcodes the same as the absence of barcodes.
3. Barcode detection is not rotation invariant.
4. Some outputs may be highly erroneous due to rotations and different lighting conditions.

## Future work-
* Integration with deep learning model(eg.-YOLO,Faster RCNN) could improve the bounding box predictions.
* For accurate barcode detection, the use of more robust libraries could handle rotation invariance. 
