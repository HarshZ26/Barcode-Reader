from init import *
'''Function to detect contours and crop the images.'''
def preprocess(img):
  # Pass the image to the function
  image = img.copy()
  # To make image more suitable to detect contours(to increase or decrease seprartion between two objects)
  image = cv2.erode(image, None, iterations=4)
  image = cv2.dilate(image, None, iterations=1)

  # Convert the image to grayscale and then to binary
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7, 7), 0)
  #Here, Adaptive thresholding is used to handle different lighting conditions
  binary = cv2.adaptiveThreshold(blurred, 
                            255, 
                            cv2.ADAPTIVE_THRESH_MEAN_C, 
                            cv2.THRESH_BINARY, 
                            31, 
                            10)
  #Invert the image for contour detection
  inverted_binary = ~binary
  # Find the contours and store them in a list
  contours, _ = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
  
  # Print the total number of contours that were detected
  print('Total number of contours detected: ' + str(len(contours)))

  # Predict bounding boxes around all contours and store the cropped sections of images in a list
  img_lis = []
  bbox_lis = []
  for c in contours:
    x, y, w, h = cv2.boundingRect(c)
      # To filter out tiny boxes(variable that can be changed)
    if (cv2.contourArea(c)) > 10000:
      img_lis.append(img[y:(y+h),x:(x+w)])
      bbox_lis.append([x,y,w,h])
  
  result,b_values = detect_bcode(img_lis)
  final_img = post_process(img,bbox_lis,result)
  return final_img,b_values

'''Function to detect barcodes from the cropped images.'''
def detect_bcode(lis):
    detect_lis = dict()
    dta = []
    for img in lis:
        result = decode(img)
        #Here need to add the condition for QR code since we are detecting only barcodes.
        if len(result)==0 or result[0][1]=="QRCODE":
            dta.append([np.reshape(np.array([0,0,0,0]),(1,-1,2)),[0,0,255]])
        else:
            if result[0][0] in detect_lis.keys(): # to check occurance of barcode in the image
                detect_lis[result[0][0]]+=1
            else:
                detect_lis[result[0][0]]=1
            points = np.array(result[0][3])
            points = np.reshape(points, (1,-1,2))
            dta.append([points,[255,0,0]])
    return dta,detect_lis

'''This Function applies the non-maximal suppression to remove overlapping bounding boxes
 and draws the bounding boxes and contours on the image.'''
def post_process(img,inp,op):
    indices = non_max_suppression(inp)
    for i,item in enumerate(op):
        if i in indices:
          points =item[0]+np.reshape(np.array([[inp[i][0],inp[i][1]]]),(-1,2))
          if np.sum(item[0])==0:
            cv2.rectangle(img,(inp[i][0],inp[i][1]), (inp[i][0]+inp[i][2],inp[i][1]+inp[i][3]), item[1], 5)
          else:
            img = cv2.polylines(img, points.astype(int), True, item[1], 5)
            cv2.rectangle(img,(inp[i][0],inp[i][1]), (inp[i][0]+inp[i][2],inp[i][1]+inp[i][3]), (0,0,0), 5)
    return img

'''This function applies non-maximal suppression to remove overlapping bounding boxes.'''
def non_max_suppression(boxes, probs=None, Thresh=0.6):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # convert to float type if boxes are int type
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # list to store final indices
    pick = []
    # Extract the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0]+boxes[:, 2]
    y2 = boxes[:, 1]+boxes[:, 3]
    # Here sorting is baed on area of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.sort(area)
    # if probabilities are provided, sort on them instead(if YOLO model is used)
    if probs is not None:
        idxs = probs
    # sort the indexes
    idxs = np.argsort(idxs)
    # Apply the NMS algorithm
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap(IOU)
        overlap = (w * h) / area[idxs[:last]]

        # Remove the overlapping bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > Thresh)[0])))

    # Return the indices of the picked bounding boxes
    return pick