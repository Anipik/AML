from tflearn.data_utils import load_image ,resize_image
paths = ["train/c0","train/c1","train/c2","train/c3","train/c4","train/c5","train/c6","train/c7","train/c8","train/c9"]
path1 =  ["train1/c0","train1/c1","train1/c2","train1/c3","train1/c4","train1/c5","train1/c6","train1/c7","train1/c8","train1/c9"]
import os

# i=0
# for path in paths:
# 	for filename in os.listdir(path):
# 		print(filename)
# 		img = load_image(os.path.join(path, filename))
# 		resize_image(img, 32, 32, out_image=os.path.join(path1[i],filename))
# 	i=i+1
path = "test"
i=0
for filename in os.listdir(path):
	img = load_image(os.path.join(path,filename))
	resize_image(img,32,32,out_image=os.path.join(path,filename))
	i=i+1
	print(i)