import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os, time
from tabulate import tabulate
import random, datetime
from pallet_n_classnames import pallet_ADE20K, pallet_cityscape, pallet_VOC, pallet_mine, pallet_vistas
from IPython.display import clear_output
from models import use_customdropout, num_of_classes
import matplotlib as mpl
from scipy import signal


mpl.rcParams['figure.dpi'] = 300
import tensorflow as tf
if int(str(tf.__version__)[0]) == 1:
    from keras import backend as K
    from keras.callbacks import Callback
    from keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D
if int(str(tf.__version__)[0]) == 2:
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D
    
use_mydropout = use_customdropout()

from layers import DropBlock2D
if use_mydropout == True:
        from layers import DropBlock2D, Dropout, SpatialDropout2D
elif use_mydropout == False:
    if int(str(tf.__version__)[0]) == 1:
        from keras.layers import Dropout, SpatialDropout2D
    if int(str(tf.__version__)[0]) == 2:
        from tensorflow.keras.layers import Dropout, SpatialDropout2D
        
   
    
num_class = num_of_classes()
# creat record file to write custom scaleres
weight_name = 'Efficent_pet_203_clf'
log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M") + weight_name
logdir='/home/user01/data_ssd/Talha/pannuke/pan_final/logs/{}/'.format(log_name)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

class PlotLearning(Callback):
    
    def __init__(self, table=None):
        self.table = table
    
    def on_train_begin(self, epoch, logs={}):
        self.i = 0
        self.x = []
        self.clf_losses = []
        self.clf_val_losses = []
        self.seg_losses = []
        self.seg_val_losses = []
        self.inst_losses = []
        self.inst_val_losses = []
        self.acc = []
        self.val_acc = []
        self.iou = []
        self.val_iou = []
        self.ioup = []
        self.val_ioup = []
        self.lr = []
        self.dropout = []
        self.fig = plt.figure()
        self.t_op = 0.
        self.v_op = 0.
        self.result = []
        self.logs = []
        

    def on_epoch_end(self, epoch, logs={}):
        
        self.metric1 = self.model.metrics_names[1] # clf_loss
        self.metric2 = self.model.metrics_names[2] # seg_loss
        self.metric3 = self.model.metrics_names[3] # inst_loss
        self.metric4 = self.model.metrics_names[4] # clf_accuracy
        # self.metric5 = self.model.metrics_names[5] # seg_iou
        # self.metric6 = self.model.metrics_names[6] # inst_iou
        self.logs.append(logs)
        self.x.append(self.i)
        # clf and seg loss
        self.clf_losses.append(logs.get(self.metric1))
        self.clf_val_losses.append(logs.get('val_' + self.metric1))
        self.seg_losses.append(logs.get(self.metric2))
        self.seg_val_losses.append(logs.get('val_' + self.metric2))
        self.inst_losses.append(logs.get(self.metric3))
        self.inst_val_losses.append(logs.get('val_' + self.metric3))
        # accuracy and iou
        self.acc.append(logs.get(self.metric4))
        self.val_acc.append(logs.get('val_' + self.metric4))
        # self.iou.append(logs.get(self.metric5))
        # self.val_iou.append(logs.get('val_' + self.metric5))
        # self.ioup.append(logs.get(self.metric6))
        # self.val_ioup.append(logs.get('val_' + self.metric6))
        
        
        self.lr.append(float(K.get_value(self.model.optimizer.lr)))
        
        try:
            if len(self.model.layers) > 10:
                for layer in self.model.layers:
                    if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                        self.dropout.append(K.get_value(layer.rate))
                        break
            else:
                for layer in self.model.layers[-2].layers:
                    if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                        self.dropout.append(K.get_value(layer.rate))
                        break
        except: # if no dropout layer is present
            self.dropout.append(0)
            
        self.i += 1
        f, ax = plt.subplots(2, 3, figsize = (10,7), sharex=False)
        
        clear_output(wait=True)
        #yhat=signal.savgol_filter(y,53, 3) for smooting plotting
        #ax1.set_yscale('log')
        ax[0,0].plot(self.x, self.clf_losses, 'g', label=self.metric1)
        ax[0,0].plot(self.x, self.clf_val_losses, 'r', label='val_' + self.metric1)
        ax[0,0].legend()
        ax[0,1].plot(self.x, self.seg_losses, 'g', label=self.metric2)
        ax[0,1].plot(self.x, self.seg_val_losses, 'r', label='val_' + self.metric2)
        ax[0,1].legend()
        ax[1,0].plot(self.x, self.inst_losses, 'blue', label=self.metric3)
        ax[1,0].plot(self.x, self.inst_val_losses, 'orange', label='val_' + self.metric3)
        ax[1,0].legend()
        
        ax[1,1].plot(self.x, self.acc, 'blue', label=self.metric4)
        ax[1,1].plot(self.x, self.val_acc, 'orange', label='val_' + self.metric4)
        ax[1,1].legend()
        
        ax[0,2].plot(self.x, self.lr, 'm', label='Learning Rate')
        ax[0,2].legend()
        
        ax[1,2].plot(self.x, self.dropout, 'c', label='Dropout')
        ax[1,2].legend()
        
        plt.show();
        
        self.t_op = [logs.get('loss'), logs.get(self.metric1), logs.get(self.metric2), logs.get(self.metric3),
                      logs.get(self.metric4)]
        
        self.v_op = [logs.get('val_loss'), logs.get('val_' + self.metric1), logs.get('val_' + self.metric2),
                      logs.get('val_' + self.metric3), logs.get('val_' + self.metric4)],
        # wite in tensorboard
        tf.summary.scalar('Hyperparameter: Learning rate', data=self.lr[-1], step=epoch)
        tf.summary.scalar('Hyperparameter: Dropout', data=self.dropout[-1], step=epoch)
        # print after every epoch
        self.result = np.concatenate((np.asarray(self.model.metrics_names).reshape(-1,1), 
                                      np.round(np.asarray(self.t_op).reshape(-1,1),4), np.round(np.asarray(self.v_op).reshape(-1,1),4)), 1)
        result = tabulate(np.ndarray.tolist(self.result), headers = ["Metric", "Train_Value", "Val_value"], tablefmt="github")
        print(result)
        tf.summary.text('Hyperparameter: resutls after epoch', result, step=epoch, description=None)
        tf.summary.text('Hyperparameter: values', self.table, step=epoch, description=None)

        
class PredictionCallback(Callback):
        '''
        Decalre Arguments Input Here
        '''
        def __init__(self, path_train, im_height, im_width):
            self.path_train = path_train
            self.im_height = im_height
            self.im_width = im_width
            self.j = 0
        def on_epoch_end(self, epoch, logs={}):
            random.seed(self.j)
            path_images = self.path_train + '/images/images/'
            n = random.choice([x for x in os.listdir(path_images) 
                                 if os.path.isfile(os.path.join(path_images, x))]).rstrip('.jpg')
            
            img = cv2.imread(self.path_train+'/images/images/'+ n + '.jpg')
            if img is None:
                  img = cv2.imread(self.path_train+'/images/images/' + n + '.png')
                  
            img = cv2.resize(img, (self.im_width, self.im_height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            y_pred = self.model.predict(img[np.newaxis,:,:,:]/255)
            if y_pred.shape[-1] < 2:
                y_pred = sigmoid_activation(y_pred)
            else:
                y_pred = softmax_activation(y_pred)
            y_pred = (y_pred > 0.5).astype(np.uint8)
            y_pred = np.argmax(y_pred.squeeze(), 2)
            y_pred = tf_gray2rgb(y_pred, pallet_mine)
            
            '''
            if y_pred.shape[-1] < 2:
                y_pred = sigmoid_activation(y_pred)
                y_pred = (y_pred > 0.5).astype(np.uint8)
            else:
                y_pred = softmax_activation(y_pred)
                y_pred = (y_pred > 0.5).astype(np.uint8)
                y_pred = np.argmax(y_pred.squeeze(), 2)
                y_pred = tf_gray2rgb(y_pred, pallet_mine)
            
            y_pred = np.squeeze(y_pred)
            '''
            
            gt = cv2.imread(self.path_train+'/masks/masks/'+ n + '.png', 0)
            if gt is None:
                  gt = cv2.imread(self.path_train+'/masks/masks/' + n + '.jpg', 0)
            #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, (self.im_width, self.im_height), interpolation=cv2.INTER_NEAREST)
            self.j += 1
            
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,4), sharex=False)
            f.suptitle('Pred after {} Epoch(s)'.format(epoch+1))
            clear_output(wait=True)
            
            ax1.imshow(img)
            ax1.axis("off")
            
            ax2.imshow(gt)
            ax2.axis("off")
            
            ax3.imshow(y_pred)
            ax3.axis("off")
            
            plt.show();



class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule, initial_lr, lr_decay, total_epochs, drop_epoch, power):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.total_epochs = total_epochs
        self.drop_epoch = drop_epoch
        self.power = power
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        if self.schedule == 'step_decay':
            self.schedule = step_decay
        if self.schedule == 'polynomial_decay':
            self.schedule = polynomial_decay
        if self.schedule == 'K_decay':
            self.schedule = K_decay
            
        lr = self.initial_lr
        if lr is None:
            # Get the current learning rate from model's optimizer.
            lr = float(K.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.lr_decay, self.drop_epoch, self.total_epochs, self.power)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}".format(epoch+1, scheduled_lr))

class CustomDropoutScheduler(Callback):
    def __init__(self, schedule, dropout_after_epoch):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = np.append(schedule, [1], axis=0)
        self.DAE = np.append(dropout_after_epoch, [100], axis=0)

    def on_epoch_begin(self, epoch, logs=None):
        if len(self.model.layers) > 10: # for handeling the multi gpu strategy
            for i in range(len(self.DAE)):
                if epoch >= self.DAE[i] and epoch <= self.DAE[i+1]:
                    print('Updating Dropout Rate...')
                    for layer in self.model.layers:
                        if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
                    break
            for layer in self.model.layers:
                if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                    print('Epoch %05d: Dropout Rate is %6.4f' % (epoch+1, K.get_value(layer.rate)))
                    print(60*'=')
                    break
        else:
            for i in range(len(self.DAE)):
                if epoch >= self.DAE[i] and epoch <= self.DAE[i+1]:
                    print('Updating Dropout Rate*...')
                    for layer in self.model.layers[-2].layers:
                        if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
                    break
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Dropout) or isinstance(layer, SpatialDropout2D)  or isinstance(layer, DropBlock2D):
                    print('Epoch %05d: Dropout Rate* is %6.4f' % (epoch+1, K.get_value(layer.rate)))
                    print(60*'=')
                    break

class CustomKernelRegularizer(Callback):
    '''
    def __init__(self, schedule, dropout_after_epoch):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.DAE = dropout_after_epoch
    '''
    def on_epoch_begin(self, epoch, logs=None):
        if len(self.model.layers) > 10:
            for layer in self.model.layers:
                if isinstance(layer, Conv2D) == True:
                    print('Conv2D: kernel_regularizer = %6.2e' % (layer.kernel_regularizer.l2))
                    break
            for layer in self.model.layers:
                if isinstance(layer, SeparableConv2D) == True:
                    print('SeparableConv2D: kernel_regularizer = %6.2e' % (layer.depthwise_regularizer.l2))
                    print(60*'_')
                    break
        else:
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Conv2D) == True:
                    print('Conv2D: kernel_regularizer = %6.2e' % (layer.kernel_regularizer.l2))
                    break 
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, SeparableConv2D) == True:
                    print('SeparableConv2D: kernel_regularizer = %6.2e' % (layer.depthwise_regularizer.l2))
                    print(60*'_')
                    break    

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())
        
    def on_epoch_begin(self, epoch, logs=None):
        print(60*'=')
        print("Epoch %05d: Learning rate is %6.2e"  % (epoch+1, K.get_value(self.model.optimizer.lr)))
        
    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()
            
    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps => int(Epoch * num_images/Batch_size)
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
                             int(warmup_epoch * num_images/Batch_size)
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps, i.e. 
                              number of iterations (total_epochs * total_iamges/batch_size)
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0}) 
                              to set at 1 epoch set to -> int(warmup_epoch * num_images/Batch_size)
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []
        self.x = np.arange(total_steps)
        
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        
    def on_epoch_end(self, epoch, logs={}):
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.xlabel('Iterations')
        plt.ylabel('Magnitude')
        #print('\nBatch %05d: setting learning rate to %s.' % (self.global_step + 1, lr))
#%%   
def sigmoid_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.sigmoid(pred)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def softmax_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.softmax(pred, axis=-1)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def tf_gray2rgb(gray_processed, pallet):
    
    pallet = tf.convert_to_tensor(pallet)
    w, h = gray_processed.shape
    gray = gray_processed[:,:,np.newaxis]
    gray = tf.image.grayscale_to_rgb((tf.convert_to_tensor(gray)))
    gray = tf.cast(gray, 'int32')
    unq = np.unique(gray_processed)
    rgb = tf.zeros_like(gray, dtype=tf.float64)
    
    for i in range(len(unq)):
        clr = pallet[:,unq[i],:]
        clr = tf.expand_dims(clr, 0)
        rgb = tf.where(tf.not_equal(gray,unq[i]), rgb, tf.add(rgb,clr))
        
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        rgb = sess.run(rgb)
    else:
        rgb = rgb.numpy()
    return rgb

def step_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    drop = lr_decay
    epochs_drop = drop_epoch
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def polynomial_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    lrate = initial_lrate * math.pow((1-(epoch/Epoch)),power)
    return lrate
x = np.arange(0,50)# current epoch 
Epoch  = 50
k = 0.4
N = 1
inint_lr = 0.002
final_lr = 0

def K_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power, Le=1e-7, N=4, k=3):
    t = epoch
    L0 = initial_lr
    T = Epoch
    lr = (L0 - Le) * (1 - t**k / T**k)**N + Le
    return lr