import cv2
import numpy as np
import os
import sys
def crop_image(img, dir_name, size):
	h, w = size
	i = 0
	for x in range(0, img.shape[1], w):
		for y in range(0, img.shape[0], h):
			crop_img = cv2.resize(img[y:y+h, x:x+w], (h, w))
			h_ = min(h, img.shape[0] - y)
			w_ = min(w, img.shape[1] - x)
			#crop_img = img[y:y+h_, x:x+w_]
			crop_img[0:h_, 0:w_] = img[y:y+h_, x:x+w_]
			#crop_img[h_:,:] = img[y+h_-1, :]
			cv2.imwrite('%s/%d_%d.jpg' % (dir_name, y, x), crop_img)
			#cv2.imwrite('%s/%d.jpg' % (dir_name, i), crop_img)
			i += 1
def crop_images(imgs, dir_name, size):
	h, w = size
	i = 0
	for img in imgs:
		for x in range(0, img.shape[1], w):
			for y in range(0, img.shape[0], h):
				crop_img = np.zeros((h, w, 3))
				h_ = min(h, img.shape[0] - y)
				w_ = min(w, img.shape[1] - x)
				crop_img[0:h_, 0:w_] = img[y:y+h_, x:x+w_]
				cv2.imwrite('%s/%d_%d.jpg' % (dir_name, y, x), crop_img)
				#cv2.imwrite('%s/%d.jpg' % (dir_name, i), crop_img)
				i += 1

if __name__ == '__main__':
	argc = len(sys.argv)
	if(argc != 2 and argc != 1):
		print('[usage]: python3 %s [image_name]' % sys.argv[0])
		exit(0)
	dir_name = 'original'
	os.makedirs(dir_name, exist_ok=True)
	if(argc == 2):
		img_name = sys.argv[1]
		img = cv2.imread(img_name)
		crop_image(img, dir_name, (512, 512))
	else:
		imgs = os.listdir('./')
		imgs = [cv2.imread(path) for path in imgs]
		crop_images(imgs, dir_name, (512, 512))
	#crop_image(Restore_img, 'Restore_cropped', (300, 300))
