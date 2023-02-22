from init import *
from bcode import *

if __name__=="__main__":
  for item in os.listdir(r"barcode\Input_data"):
    if item.endswith(".jpg"):
      image = cv2.imread(r"barcode\Input_data\{}".format(item))
      Output,b_values = preprocess(image)
      print("The detected barcodes and their unique count.")
      pprint.pprint(b_values)
      cv2.imwrite(os.path.join('barcode\Results','Output_'+item), Output)
  cv2.imshow('Output', Output)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
