import re
import random
import numpy as np
import cv2
import os.path
import shutil
from glob import glob

# retrive image data from disk
# returns image in RGB format
def readImage(img_fname):
  img_path = os.path.join(DATA_DIR,img_fname.strip())
  img = cv2.imread(img_path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# adds a random shadow to a given image
def addShadow(img):
  # pick a random shadow coloration
  shadow_shade = np.random.randint(60,120)
  # convert image to YUV space to get the luma (brightness) channel
  y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
  y = y.astype(np.int32)
  # create mask image with same shape as input image
  #mask = np.zeros(y.shape, dtype=np.int32)
  # compute a random line in slope, intercept form
  # random x1,x2 values (y1=0, y2=height)
  x1 = np.random.uniform() * y.shape[1]
  x2 = np.random.uniform() * y.shape[1]
  slope = float(y.shape[0]) / (x2 - x1)
  intercept = -(slope * x1)
  # assign pixels of mask below line
  for j in range(y.shape[0]):
      for i in range(y.shape[1]):
          if j > (i*slope)+intercept:
              y[j,i] -= shadow_shade
  # apply mask
  #y += mask
  # ensure values are within uint8 range to avoid artifacts
  y = np.clip(y, 0,255).astype(np.uint8)
  # convert back to RGB
  return cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YCrCb2RGB)

# adjust brightness of given image (img) by multiplyling V (brightness)
def adjustBrightness(img, m):
  h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
  v = np.clip(v * m, 0, 255).astype(np.uint8)
  return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2RGB)


if __name__ == '__main__':
	data_folder = "./data/data_road/training"

	image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
	label_paths = {
	    re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
	    for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
	background_color = np.array([255, 0, 0])

	for image_file in image_paths:
		print("processing: {0}".format(image_file) )
		gt_image_file = label_paths[os.path.basename(image_file)]
		img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
		aug = [addShadow(img), adjustBrightness(img, .75), adjustBrightness(img, .5), adjustBrightness(img, .25)]
		for idx, augimg in enumerate(aug):
			cv2.imwrite("{0}_aug{2}{1}".format(*(list(os.path.splitext(image_file))+[idx])), augimg)
			shutil.copy(gt_image_file, "{0}_aug{2}{1}".format(*(list(os.path.splitext(gt_image_file))+[idx])))
