#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 yuengdelahoz <yuengdelahoz@TAOs-Macbook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
"""

import sys, os, shutil, cv2, shutil, pickle, traceback
import numpy as np
from collections import namedtuple
import threading
import cv2
import shutil

def clear_folder(name):
	if os.path.isdir(name):
		try:
			shutil.rmtree(name)
		except:
			 pass
			# print(name,'could not be deleted')
			# traceback.print_exc(file=sys.stdout)

def create_folder(name,clear_if_exists = True):
	if clear_if_exists:
		clear_folder(name)
	try:
		os.makedirs(name)
		return name
	except:
		pass
		# print(name,'could not be created')
		# traceback.print_exc(file=sys.stdout)

def generate_training_validation_test_sets():
	Data = namedtuple('dataset',['training_set','testing_set','validation_set'])
	color_imgs = os.listdir('images/color')
	np.random.shuffle(color_imgs)
	for i,img in enumerate(color_imgs):
		if not img.endswith('png'):
			del color_imgs[i]
			print (i,img)
	sz = len(color_imgs)

	train = color_imgs[:int(sz*0.7)]
	test = color_imgs[int(sz*0.7):int(sz*0.95)]
	validation = color_imgs[int(sz*0.95):]
	dataset = Data(training_set=train, testing_set=test, validation_set=validation)
	pickle.dump(dataset._asdict(),open('dataset.pickle','wb'))

def cropImages():
	create_folder('dataset/input')
	create_folder('dataset/label')
	cnt = 0
	for img in os.scandir('images/color'):
		if img.name.endswith('.png'):
			print('Cropping',img.name)
			color = cv2.imread(img.path,cv2.IMREAD_COLOR)
			label = cv2.imread(img.path.replace('color','label'),cv2.IMREAD_GRAYSCALE)
			for shift in range(0,80,10): # 8 crops
				new_color = color[:,shift:240+shift]
				new_label = label[:,shift:240+shift]
				name = 'img_{:010}.png'.format(cnt)
				cv2.imwrite('dataset/input/'+name,new_color)
				cv2.imwrite('dataset/label/'+name,new_label)
				cnt +=1
	print('Done cropping')

def createSuperLabels():
	path1 = create_folder('images/superlabel/')
	"""
	There are 240x240 = 57600 pixels, so every superpixels (6 in total) has 57600/6=9600 pixels (120x80)
	The resolution of each superlabel is 8x8 pixels
	img[rows,cols]
	img[0,0] = 0 (black)
	img[0,0] = 255 (white)
	"""
	cnt = 0
	for img in os.scandir('images/label'):
		# if np.random.randint(0,2) > 0:
			# continue
		print('Creating superlabel for',img.path,end='\r')
		sys.stdout.write("\033[K")
		label = cv2.imread(img.path,cv2.IMREAD_GRAYSCALE)
		superlabel = list() # empty list where to append Superpixels
		for idj,j in enumerate(range(0,240,120)): # 2 superlabels in the height direction
			for idk,k in enumerate(range(0,240,80)): # 3 superlabels in the width direction
				blob = label[j:j+120,k:k+80]# img[rows,cols]
				pix_sum = np.sum(blob)
				if pix_sum > 0.95 * 255 * 9600 : # if pixel values sum is more than 90% white.
					superlabel.append(0) # mark superlabel as a ZERO
				else:
					superlabel.append(1) # mark superlabel as a ONE
		cnt +=1
		# if cnt == 100:
			# break
		np.save(path1+img.name.replace('.png',''),np.array(superlabel))
	print('Done creating superlabels,',cnt,'superlabels were created')

def paintImagesAll():
	path1 = create_folder('painted_images/color/')
	path2 = create_folder('painted_images/label/')
	path3 = create_folder('painted_images/color_label/')
	"""Iterate over original image (color) and paint (red blend) the superpixels that were identified as being part of the floor by the neural network"""
	cnt = 0
	for img in os.scandir('images/superlabel'):
		if img.name.endswith('.npy'):
			cnt +=1
			print('painting',img.name,end='\r')
			sys.stdout.write("\033[K")
			color = cv2.imread(img.path.replace('superlabel','input').replace('npy','png'),cv2.IMREAD_COLOR)
			label = cv2.imread(img.path.replace('superlabel','label').replace('npy','png'),cv2.IMREAD_GRAYSCALE)
			superlabel = np.load(img.path)
			pos = 0
			for sv in range(0,240,120): # 2 superpixels in the height direction
				for sh in range(0,240,80): # 3 superpixels in the width direction
					cv2.rectangle(color,(sh,sv),(sh+80,sv+120),(255,0,0),2)
					if superlabel[pos]==1:
						blob = color[sv:sv+120,sh:sh+80] 
						red =np.zeros(blob.shape)
						red[:,:,2] = np.ones(red.shape[0:2])*255
						color[sv:sv+120,sh:sh+80] = blob*0.5 + 0.5*red
					pos +=1
			color_label = np.concatenate((color,cv2.cvtColor(label,cv2.COLOR_GRAY2BGR)), axis=1)
			cv2.imwrite(path1+img.name.replace('npy','png'),color)
			cv2.imwrite(path2+img.name.replace('npy','png'),label)
			cv2.imwrite(path3+img.name.replace('npy','png'),color_label)
	print('Done painting images',cnt,'images were painted')

def paintImage(image,superlabel):
	paintedImg = image.copy()
	pos = 0
	for sv in range(0,240,8): # 12 superpixels in the height direction
		for sh in range(0,240,8): # 12 superpixels in the width direction
			if superlabel[pos]==1:
				red =np.zeros(image[sv:sv+8,sh:sh+8].shape)
				red[:,:,2] = np.ones(red.shape[0:2])*255
				paintedImg[sv:sv+8,sh:sh+8] = image[sv:sv+8,sh:sh+8]*0.5 + 0.5*red # 90% origin image, 10% red
			pos +=1
	return paintedImg

def paintBatch(inputBatch,outputBatch,output_folder):
	print('Painting images in the batch')
	output_folder = 'Painted_Images/'+output_folder 
	create_folder(output_folder)
	for i,(img,suplbl) in enumerate(zip(inputBatch,outputBatch)):
		pimg = paintImage(img,suplbl)
		name = '{}/img_{:02}.png'.format(output_folder,i)
		cv2.imwrite(name,pimg)

class PainterThread (threading.Thread):
	def __init__(self,gt_input,gt_labels,net_output,output_folder='Training'):
		threading.Thread.__init__(self)
		self.input = gt_input
		self.gt_superlabels = gt_labels
		self.net_superlabels= net_output
		self.folder = output_folder
	def run(self):
		path1 = create_folder('painted_images/'+self.folder+'/color_gt/')
		path2 = create_folder('painted_images/'+self.folder+'/color_net/')
		path3 = create_folder('painted_images/'+self.folder+'/color_gt_and_net/')
		font = cv2.FONT_HERSHEY_SIMPLEX
		for idx,(color_gt,superlabel_gt,superlabel_net) in enumerate(zip(self.input,self.gt_superlabels,self.net_superlabels)):
			img_name = 'image_{:.2f}.png'.format(idx)
			i = 0
			color_net = color_gt.copy()
			pos = 0
			for sv in range(0,240,120): # 2 superpixels in the height direction
				for sh in range(0,240,80): # 3 superpixels in the width direction
					# Ground Truth
					if superlabel_gt[pos]==1:
						blob = color_gt[sv:sv+120,sh:sh+80] 
						red =np.zeros(blob.shape)
						red[:,:,2] = np.ones(red.shape[0:2])*255
						color_gt[sv:sv+120,sh:sh+80] = blob*0.5 + 0.5*red

					if superlabel_net[pos]==1:
						blob = color_net[sv:sv+120,sh:sh+80] 
						red =np.zeros(blob.shape)
						red[:,:,2] = np.ones(red.shape[0:2])*255
						color_net[sv:sv+120,sh:sh+80] = blob*0.5 + 0.5*red
					pos +=1
					cv2.rectangle(color_gt,(sh,sv),(sh+80,sv+120),(255,0,0),2)
					cv2.rectangle(color_net,(sh,sv),(sh+80,sv+120),(255,0,0),2)

			cv2.putText(color_gt,'GT',(10,15), font, 0.4,(0,255,0),1,cv2.LINE_AA)
			cv2.putText(color_net,'NET',(10,15), font, 0.4,(0,255,0),1,cv2.LINE_AA)

			color_gt_net = np.concatenate((color_gt,color_net), axis=1)
			cv2.imwrite(path1+img_name,color_gt)
			cv2.imwrite(path2+img_name,color_net)
			cv2.imwrite(path3+img_name,color_gt_net)
		print('Done painting images in batch')

def calculateMetrics(GroundTruthBatch, OutputBatch):
	''' This method calculates Accuracy, Precision, and Recall
		Relevant items = Superpixels that represent Objects on the floor
		TP = True Positive - Superpixels that were correctly classified as part of the object
		TP = True Positive - Superpixels that were correctly classified as part of the object
		TN = True Negative - Superpixels that were correctly classified as NOT part of the object
		FP = False Positive - Superpixels that were INcorrectly classified as part of the object
		FN = False Negative - Superpixels that were INcorrectly classified as NOT part of the object.

		Accuracy = (TP + TN)/(TP + TN + FP +FN)
		Precision = TP/(TP+FP)
		Recall = TP/(TP + FN)
	'''
	Accuracy = []
	Precision = []
	Recall = []
	for i in range(len(GroundTruthBatch)):
		GT = 2*GroundTruthBatch[i]
		NET = OutputBatch[i]
		RST = GT - NET
		TP,TN,FP,FN = 0,0,0,0
		for v in RST:
			if v == 0:
				TN += 1
			elif v == 1:
				TP += 1
			elif v == -1:
				FP += 1
			elif v == 2:
				FN +=1
		# print ('TP',TP,'TN',TN,'FP',FP,'FN',FN)
		acc = (TP + TN)/(TP + TN + FP +FN)
		if TP + FP !=0:
			prec = TP/(TP + FP)
			Precision.append(prec)
		if TP + FN !=0:
			rec = TP/(TP + FN)
			Recall.append(rec)
		Accuracy.append(acc)
	return np.mean(Accuracy) if len(Accuracy)>0 else 0,np.mean(Precision)if len(Precision)>0 else 0,np.mean(Recall)if len(Recall)>0 else 0

def is_model_stored(top):
	try:
		model_files = os.listdir('Models/'+top)
		model_stored = False
		for mf in model_files:
			if 'model' in mf:
				model_stored = True
				break
		return model_stored
	except:
		return False

def generate_new_binary_dataset_for_objects_on_the_floor():
	ipath = create_folder('Dataset/Images3/input/')
	lpath = create_folder('Dataset/Images3/label/')
	for npy in os.scandir('Dataset/Images/superlabel'):
		if npy.name.endswith('.npy'):
			shutil.copy('Dataset/Images/input/'+npy.name.replace('npy','png'),ipath+npy.name.replace('npy','png'))
			sp = np.load(npy.path)
			if sum(sp) > 0:
				label = [1,0]
			else:
				label = [0,1]
			np.save(lpath+npy.name,label)
			sys.stdout.write("\033[K")
			print('generating new label for',npy.name,end='\r')
	print('\nDone')

def paint_all_images_with_text():
	path = create_folder('Dataset/painted_images/')
	font = cv2.FONT_HERSHEY_SIMPLEX
	cnt = 0
	for img in os.scandir('Dataset/Images/input'):
		if img.name.endswith('png'):
			color = cv2.imread(img.path)
			label = np.load(img.path.replace('input','label').replace('png','npy'))
			if np.array_equal(label,[1,0]):
				cv2.putText(color,'YES',(100,100), font, 1,(0,0,255),2,cv2.LINE_AA)
			elif np.array_equal(label,[0,1]):
				cv2.putText(color,'NO',(100,100), font, 1,(255,0,0),2,cv2.LINE_AA)
			cv2.imwrite(path+img.name,color)
			print('painting',img.name,np.array_equal(label,[1,0]),end='\r')
			sys.stdout.write("\033[K")
			if cnt == 100:
				break
			cnt +=1
	print('Done')

def log_data(folder_path,data,mode='a'):
	try:
		if type(data) is dict:
			with open(folder_path+'/README.txt',mode) as f:
				for k,v in sorted(data.items()):
					f.write(k + " : " + str(v) + '\n')
		else:
			with open(folder_path+'/README.txt',mode) as f:
				data = str(data)
				f.write(data)
	except:
		traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
	# createSuperLabels()
	paintImagesAll()

