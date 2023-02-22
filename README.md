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
* For barcode detection, the use of more robust libraries could handle achieve rotation invariance. 
