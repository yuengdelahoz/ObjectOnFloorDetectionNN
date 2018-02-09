import tensorflow as tf
import numpy as np
import cv2
import time
from .Layers import Layer
from Dataset.DataHandler import DataHandler
from Utils import utils
import operator
import functools
import cv2
import os
import shutil,sys,traceback
from pprint import pprint

class Network:
	def __init__(self):
		# Read Dataset
		self.dataset = None
		self.name = None
		input('\nRemember to log the features you are using with this neural network. Press any key to continue.')

	def initialize(self,topology):
		try:
			tf.reset_default_graph()
		except:
			pass
		self.x = tf.placeholder(tf.float32, shape =[None,240,240,3],name='input_images')
		self.y = tf.placeholder(tf.float32, shape = [None,6],name='labels')
		self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')

		enabled_features = '+Floor Detection, +Early Stop, +FN Weights'
		self.layers = {'L0':'Network Topology ({})'.format(enabled_features)}
		print('Enabled features in this session',enabled_features)
		if topology == 'topology_01':
			self.topology1()
		elif topology == 'topology_02':
			self.topology2()
		elif topology == 'topology_03':
			self.topology3()
		elif topology == 'topology_04':
			self.topology4()
		elif topology == 'topology_05':
			self.topology5()

		print('\n'+self.name)

	def topology1(self):# apparently too big to handle. It gets an error saying input tensor shape too big
		self.name = 'topology_01'
		# number of parameters =
		L1 = Layer().Convolutional([5,5,3,3],self.x)# L1.output.shape = [?,120,120,3]
		L_drop = Layer().Dropout(L1.output,self.keep_prob)
		L_out = Layer(act_func = 'sigmoid').Dense([120*120*3,6],tf.reshape(L_drop.output,[-1,120*120*3]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L_drop.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology2(self): # 5 layers, 4 conv and one fully connected
		self.name = 'topology_02'
		# number of parameters = 19,949,944
		L1 = Layer(act_func = 'tanh').Convolutional([4,4,3,7],self.x,k_pool=1)# L1.output.shape = [?,500,500,7]
		L2 = Layer(act_func = 'tanh').Convolutional([5,5,7,4],L1.output)# L2.output.shape = [?,120,250,10]
		L3 = Layer(act_func = 'tanh').Convolutional([6,6,4,2],L2.output)# L3.output.shape = [?,60,125,7]
		L4 = Layer(act_func = 'tanh').Convolutional([3,3,2,3],L3.output) # L4.output.shape = [?,30,63,3]
		L_drop = Layer().Dropout(L4.output,self.keep_prob)
		L_out = Layer(act_func = 'sigmoid').Dense([30*30*3,6],tf.reshape(L_drop.output,[-1,30*30*3]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L_drop.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology3(self): # 5 layers, 4 conv and one fully connected
		self.name = 'topology_03'
		# number of parameters = 3847015
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,120,120,3]
		L2 = Layer().Convolutional([5,5,3,3],L1.output)# L2.output.shape = [?,60,60,3]
		L3 = Layer().Convolutional([5,5,3,2],L2.output)# L3.output.shape = [?,30,30,2]
		L_drop = Layer().Dropout(L3.output,self.keep_prob)
		L_out = Layer(act_func='sigmoid').Dense([30*30*2,6],tf.reshape(L_drop.output,[-1,30*30*2]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L_drop.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology4(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 8649011
		self.name = 'topology_04'
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,120,500,7]
		L2 = Layer().Convolutional([5,5,3,2],L1.output)# L2.output.shape = [?,60,500,10]
		L3 = Layer().Convolutional([6,6,2,4],L2.output,k_pool=1)# L3.output.shape = [?,60,500,7]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # L4.output.shape = [?,30,500,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output,k_pool=1) # L5.output.shape = [?,30,32,3]
		L_drop = Layer().Dropout(L5.output,self.keep_prob)
		L_out = Layer(act_func='sigmoid').Dense([30*30*3,6],tf.reshape(L_drop.output,[-1,30*30*3]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L5.__dict__})
		self.layers.update({'L6':L_drop.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology5(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 8650169
		self.name = 'topology_05'
		L1 = Layer().Convolutional([10,10,3,3],self.x,k_pool=1) # output.shape = [?,240,240,3]
		L2 = Layer().Convolutional([5,5,3,2],L1.output,k_pool=1)# output.shape = [?,240,500,2]
		L3 = Layer().Convolutional([6,6,2,4],L2.output,k_pool=1)# output.shape = [?,240,500,3]
		L4 = Layer().Convolutional([7,7,4,3],L3.output,k_pool=1) # output.shape = [?,240,250,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # output.shape = [?,120,125,3]
		L6 = Layer().Convolutional([9,9,3,2],L5.output) # output.shape = [?,60,63,3]
		L_drop = Layer().Dropout(L6.output,self.keep_prob)
		LFC = Layer().Dense([60*60*2,3600],tf.reshape(L_drop.output,[-1,60*60*2]))
		L_out = Layer(act_func='sigmoid').Dense([3600,6],LFC.output,output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers = dict()
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L5.__dict__})
		self.layers.update({'L6':L6.__dict__})
		self.layers.update({'L7':L_drop.__dict__})
		self.layers.update({'L8':LFC.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def train(self,iterations=10000,learning_rate = 1e-03):
		model_save_path = 'Models/{}/'.format(self.name)
		early_iterations_max = 200 # wait for 2000 iterations without improving to early stop
		fn_weight = 1.2

		# reading dataset
		if self.dataset is None:
			self.dataset = DataHandler().build_datasets()
		# loss function
		if fn_weight is None:
			MSE = tf.reduce_mean(tf.square(self.y - self.output))
		else:
			MSE = tf.reduce_mean(tf.square(self.y - self.output + tf.maximum((self.y - self.output) * fn_weight, 0))) #Added higher weight penalties to the false negatives
		# cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
		loss = MSE
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		lossFunc = list()

		completed_iterations = tf.Variable(0, trainable=False, name='completed_iterations')
		s1 = tf.train.Saver()

		#Early stop variables
		early_max = tf.Variable(0, trainable=False, name='early_max',dtype = tf.float32)
		early_iterations = tf.Variable(0, trainable=False, name='early_iterations')

		# Creating session and initilizing variables
		init = tf.global_variables_initializer()
		lossFunc = list()
		s2 = tf.train.Saver()

		with tf.Session() as sess:
			saver = s2
			print('Training',self.name)
			if os.path.exists(model_save_path+'model.meta'):
				print('Restoring Graph')
				try:
					s2.restore(sess,model_save_path+'model')
				except:
					sess.run(init)
					s1.restore(sess,model_save_path+'model')
					save_path = s2.save(sess,model_save_path+'model')
					print('Updating model with new Variables',save_path)
			else:
				sess.run(init)
			if early_iterations.eval() >= early_iterations_max: 
				print('Nothing to be done')
				return

			# Calculating number of trainable parameters
			lstt = tf.trainable_variables()
			acum = 0
			for lt in lstt:
				ta = lt.get_shape()
				lstd = ta.as_list()
				mult = functools.reduce(operator.mul, lstd, 1)
				acum = acum + mult
			print('Number of parameters',acum) # number of trainable parameters

			comp_iters = completed_iterations.eval()
			utils.create_folder('Models/'+self.name,clear_if_exists = not (comp_iters >0)) # clear Model folder if training has never taken place
			remaining_iterations = iterations - comp_iters
			print('Remaining Iterations:', remaining_iterations, '- Completed Iterations: ',comp_iters)

			init_time = time.time()
			last_saved_time = time.time()
			for i in range(remaining_iterations):
				start = time.time()
				batch = self.dataset.training.next_batch(50)
				normBatch = np.array([(img-128)/128 for img in batch[0]])
				labelBatch = [lbl for lbl in batch[1]]

				train_step.run(feed_dict={self.x:normBatch,self.y:labelBatch, self.keep_prob:0.5})
				if i%10==0 or i==remaining_iterations-1:
					MSE = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					print("iter {}, mean square error {}, step duration -> {:.2f} secs, time since last saved -> {:.2f} secs".format(i, MSE,(time.time()-start),time.time()-last_saved_time))
					update = comp_iters + (i+1)
					print('updating completed iterations:',sess.run(completed_iterations.assign(update)))

					batch = self.dataset.validation.next_batch(50)
					normBatch = np.array([(img-128)/128 for img in batch[0]])
					labelBatch = [lbl for lbl in batch[1]]
					results = np.round(sess.run(self.output,feed_dict={self.x:normBatch, self.y: labelBatch, self.keep_prob:1.0}))
					print("Parcial Results")
					acc,prec,rec = utils.calculateMetrics(labelBatch,results)
					print('Accuracy',acc)
					print('Precision',prec)
					print('Recall',rec)

					early_current = acc *0.2 + prec*0.4 + rec*0.4
					e_max = early_max.eval()
					delta_max_early = early_current - e_max
					print('Early stop values', early_current, e_max, delta_max_early, 'early_iterations',early_iterations.eval())
					if delta_max_early > 0.001:
						model_max_path = utils.create_folder(model_save_path+'max/')
						sess.run(early_max.assign(early_current)) # update max value
						sess.run(early_iterations.assign(0)) # reset early_iterations counter
						save_path = saver.save(sess,model_max_path+'model')
						print("Model saved in file: %s" % save_path)
						print('early max after update',early_max.eval())
						utils.Painter(batch[0],batch[1],results,output_folder='Training/'+self.name).run()
						
					else:
						e_iters = early_iterations.eval()
						if e_iters >= early_iterations_max: # no improvement has been detected. STOP
							save_path = saver.save(sess,model_save_path+'model')
							break
						elif e_iters >0 and e_iters % 100==0:
							learning_rate = learning_rate*0.95 # dicrease learning rate if no improvement is detected
						sess.run(early_iterations.assign_add(1))
					save_path = saver.save(sess,model_save_path+'model')
					print("Model saved in file: %s \n\n" % save_path)
					last_saved_time = time.time()

			utils.log_data(model_save_path,self.layers,mode='w')
			utils.log_data(model_save_path,{'\nFalse Negative Penalization Weight':fn_weight}) if fn_weight is not None else None
			msg = "\nNumber of parameters = {}\nNumber of iterations = {}\nLearning rate = {}\n".format(acum,comp_iters,learning_rate)
			utils.log_data(model_save_path,msg)
			self.freeze_graph_model(sess)
			print('total time -> {:.2f} secs'.format(time.time()-init_time))

	def evaluate(self,topology=None):
		try:
			tf.reset_default_graph()
		except:
			pass

		if topology is None:
			topology = self.name

		if self.dataset is None:
			self.dataset = DataHandler().build_datasets()

		topology_path ='Models/{}/'.format(topology)
		if not os.path.exists(topology_path+'max/model.meta'):
			print('No model stored to be restored.')
			return
		print('Evaluating',topology)
		s1 = tf.train.import_meta_graph(topology_path+'max/model.meta')
		g = tf.get_default_graph()
		x = g.get_tensor_by_name("input_images:0")
		y = g.get_tensor_by_name("labels:0")
		keep_prob = g.get_tensor_by_name("keep_prob:0")
		output= g.get_tensor_by_name("superpixels:0")

		with tf.Session() as sess:
			s1.restore(sess,topology_path + 'max/model')
			print("Model restored.")

			eval_vars = {'num_of_evals':'','accuracy':'','precision':'','recall':''} # additonal variables to store evaluation results
			for name in eval_vars.keys():
				flag = True
				for v in tf.global_variables(scope='evaluation'):
					if name in v.name:
						# print('updating evaluation_variables',v)
						eval_vars[name] = v
						flag = False
						break
				if flag:
					with tf.variable_scope('evaluation'):
						eval_vars[name] = tf.Variable(0,trainable=False,name=name,dtype=tf.float32)
						sess.run(eval_vars[name].initializer)
					# print('creating evaluation_variables',eval_vars[name])

			if eval_vars['num_of_evals'].eval() > 0:
				eval_metrics = '\nEvaluation metrics\nAccuracy: {0:.2f}, Precision: {1:.2f}, Recall {2:.2f}'.format(
						eval_vars['accuracy'].eval(),
						eval_vars['precision'].eval(),
						eval_vars['recall'].eval())
				print('Training has already been completed',eval_metrics)
				return
			# Evaluating testing set
			metrics = []
			for  i in range (self.dataset.testing.num_of_images//50):
				batch = self.dataset.testing.next_batch(50)
				testImages = np.array([(img-128)/128 for img in batch[0]])
				testLabels = [lbl for lbl in batch[1]]
				results = np.round(sess.run(output,feed_dict={x:testImages,y: testLabels,keep_prob:1.0}))
				met = utils.calculateMetrics(testLabels,results)
				print ('iter',i,'Metrics',met,end='\r')
				metrics.append(met)

			metrics = np.mean(metrics,axis=0)
			eval_metrics = '\nEvaluation metrics\nAccuracy: {0:.2f}, Precision: {1:.2f}, Recall {2:.2f}'.format(metrics[0],metrics[1],metrics[2])
			print(eval_metrics)
			utils.log_data(topology_path,eval_metrics)

			sess.run(eval_vars['num_of_evals'].assign_add(1)) # mark model as already evaluated
			sess.run(eval_vars['accuracy'].assign(metrics[0])) # 
			sess.run(eval_vars['precision'].assign(metrics[1])) # 
			sess.run(eval_vars['recall'].assign(metrics[2])) # 
			tf.train.Saver().save(sess,topology_path + 'max/model')

			if results is not None:
				utils.Painter(batch[0],batch[1],results,output_folder='Testing/'+topology).run()
		try:
			tf.reset_default_graph()
		except:
			pass
		shutil.copyfile('Dataset/dataset.pickle',topology_path+'dataset.pickle')
		shutil.rmtree(topology_path+'testing_images',ignore_errors=True)
		shutil.move('painted_images/Testing/'+topology,topology_path+'testing_images')

	def freeze_graph_model(self, session = None, g = None , topology = None):
		if topology is None:
			if self.name is not None:
				topology = self.name
			else:
				print('no topology was chosen')
				return

		if not utils.is_model_stored(topology):
			print("No model stored to be restored.")
			return
		try:
			tf.reset_default_graph()
		except:
			pass
		if g is None:
			g = tf.get_default_graph()

		if session is None:
			session = tf.Session()
			topology_path ='Models/{}/max/'.format(topology)
			saver = tf.train.import_meta_graph(topology_path+'model.meta')
			saver.restore(session,topology_path + 'model')

		graph_def_original = g.as_graph_def();
		# freezing model = converting variables to constants
		graph_def_simplified = tf.graph_util.convert_variables_to_constants(
				sess = session,
				input_graph_def = graph_def_original,
				output_node_names =['input_images','keep_prob','superpixels'])
		#saving frozen graph to disk
		output_folder = utils.create_folder('Models/'+topology+'/frozen')
		if output_folder is not None:
			model_path = tf.train.write_graph(
					graph_or_graph_def = graph_def_simplified,
					logdir = output_folder,
					name = 'model.pb',
					as_text=False)
			print("Model saved in file: %s" % model_path)
		else:
			print('Output folder could not be created')
