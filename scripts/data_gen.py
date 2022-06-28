import numpy as np
import matplotlib.pyplot as plt
import cv2, math, os, re, random
import matplotlib as mpl
from skimage.color import gray2rgb
from random import randint, seed
from imgaug import augmenters as iaa
import imgaug as ia
from tensorflow.python.keras.utils.data_utils import Sequence
from skimage.color import rgb2hed, hed2rgb
from stain_deconv import her_from_rgb, hed_from_rgb, deconv_stains


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def enclose_boundry(sem_mask, instances):
    frame = np.ones(sem_mask.shape)
    frame[2:-2,2:-2] = 0
    # for nuclie who are touching the image boudry
    inst_b = np.multiply(frame, sem_mask)
    inst_b = np.add(instances, inst_b)
    _,inst_b = cv2.threshold(inst_b, 0, 1, cv2.THRESH_BINARY)
    inst_b = inst_b.astype(np.uint8)
    return inst_b


def add_to_contrast(images, random_state, parents, hooks):
    '''
    A custom augmentation function for iaa.aug library
    The randorm_state, parents and hooks parameters come
    form the lamda iaa lib**
    '''
    images[0] = images[0].astype(np.float)
    img = images
    value = random_state.uniform(0.75, 1.25)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img[0] * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return ret
    
# Define the Augmentor Sequence
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# SomeOf will only apply only specifiel augmnetations on the data like SomeOf(2, ...)
# will only apply the 2 augmentaiton form the seq. list.

sometimes = lambda aug: iaa.Sometimes(0.90, aug)  

# apply on images and masks
seq = iaa.Sequential(
    [
    # apply only 2 of the following
    iaa.SomeOf(2, [
    # apply only 1 of following
    # iaa.OneOf([
        sometimes(iaa.Fliplr(0.9)),
        sometimes(iaa.Flipud(0.9)),
        sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=0, backend="cv2")),
        sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, order=0, backend="cv2")),
        sometimes(iaa.Affine(rotate=(-25, 25), order=0, backend="cv2")),
        sometimes(iaa.Affine(shear=(-8, 8), order=0, backend="cv2")),
        sometimes(iaa.KeepSizeByResize(
                             iaa.Crop(percent=(0.05, 0.25), keep_size=False),
                             interpolation='nearest')),
        ], random_order=True),
    ], random_order=True)


def augmenter(img, sem, inst, h, frame):
        # print('shape aug')
        _aug = seq._to_deterministic() 
        img = _aug.augment_images([img]) 
        sem = _aug.augment_images([sem])
        inst = _aug.augment_images([inst])
        h = _aug.augment_images([h])
        frame = _aug.augment_images([frame])
        # this will return a list of img arrays and when we convert them to array 
        img = np.squeeze(np.asarray(img)) 
        sem = np.squeeze(np.asarray(sem)) 
        inst = np.squeeze(np.asarray(inst))
        h = np.squeeze(np.asarray(h))
        frame = np.squeeze(np.asarray(frame))
        
        return img, sem, inst, h, frame

def hist_eq(img):
    
    # ip is RGB image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(img)
    ve = cv2.equalizeHist(v)
    img_eq = cv2.merge((h,s,ve))
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_HSV2RGB)
    
    return img_eq

def make_data_list(main_dir):
    # will return the list of all the images in a dir
    # get each dir
    imgs_dir = main_dir + 'images' 
    sem_masks_dir = main_dir + 'sem_masks'
    inst_mask_dir = main_dir + 'inst_masks'
    
    # get all the files in a dir
    imgs_list = sorted(os.listdir(imgs_dir), key=numericalSort)
    imgs_paths = [os.path.join(imgs_dir, fname) for fname in imgs_list]
    
    sem_masks_list = sorted(os.listdir(sem_masks_dir), key=numericalSort)
    sem_masks_paths = [os.path.join(sem_masks_dir, fname) for fname in sem_masks_list]
    
    inst_masks_list = sorted(os.listdir(inst_mask_dir), key=numericalSort)
    inst_masks_paths = [os.path.join(inst_mask_dir, fname) for fname in inst_masks_list]
    
    
    return imgs_paths, sem_masks_paths, inst_masks_paths

def Tumor_IO(img_path, sem_mask, inst_mask, modelip_img_w, modelip_img_h, data_augment=True):
    '''
    See desdcription of Depth_Data_Generator
    '''
    img = cv2.imread(img_path, -1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = deconv_stains(img, her_from_rgb)
    
    sem = cv2.imread(sem_mask, -1)
    inst = cv2.imread(inst_mask, -1)
    if len(np.unique(sem)) == 1:# b/c  only BG is present
        sem = sem * 0
        inst = inst * 0
    # b/c the overlayed boundries might contain pixel value > 1
    _,inst = cv2.threshold(inst, 0, 1, cv2.THRESH_BINARY)
    
    if data_augment == True:
        # creates a frame with 2px at boundry being zeros
        frame = np.ones(sem.shape)
        frame[2:-2, 2:-2] = 0
        
        img, sem, inst, h, frame = augmenter(img, sem, inst, h, frame)
        # Handle rotations of masks
        # for nuclie who are touching the image boudry
        inst_b = np.multiply(frame, sem)
        inst = np.add(inst, inst_b)
        _,inst = cv2.threshold(inst, 0, 1, cv2.THRESH_BINARY)
        inst = inst.astype(np.uint8)
    # verify boundries enclosement
    # still we need to enclose boundry to be consistent in test and train time
    inst = enclose_boundry(sem, inst)
    
    if img.shape[0] != modelip_img_w:
        img = cv2.resize(img, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
        h = cv2.resize(h, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
        
    # to normalize [0, 255] pixel values to [0, 1]
    # if you are using builtin keras model then dont normalize
    img = img
    h = h 
    inst = inst[:,:, np.newaxis]
    sem = sem[:,:, np.newaxis]
    
    return img, sem, inst, h
    

class Tumor_Data_Generator(Sequence) :
  '''
    Parameters
    ----------
    imgs_paths : dir path containing right frames
    sem_masks_paths : dir path containing left frames
    batch_size : batch size 
    modelip_img_w : image width 
    modelip_img_h : image hight 
    shuffle_lr : shuffle the left-right frames on inputs of encoder of CNN or not
    data_augment : augment the data or not 
    shuffle : shuffle the data after every epoch or not
    Returns
    -------
    Tensorflow/Keras Datagenerator having both inputs and outputs.
  '''
  
  def __init__(self, img_frames, sem_masks, inst_masks, tumor_types, batch_size,
               modelip_img_w, modelip_img_h, data_augment=True, shuffle=True):
    
    self.img_frames = img_frames
    self.sem_masks = sem_masks
    self.inst_masks = inst_masks
    self.tumor_types = tumor_types
    self.batch_size = batch_size
    self.modelip_img_w = modelip_img_w
    self.modelip_img_h = modelip_img_h
    self.shuffle = shuffle
    self.data_augment = data_augment
    self.indices = np.arange(len(self.img_frames))
    self.i = 0
    
  def on_epoch_end(self):
      # shuffling the indices
      if self.shuffle == True:
          np.random.shuffle(self.indices)
          # print('\n Shuffling Data...')
      
  def __len__(self) :
    # getting the total no. of iterations in one epoch
    return (np.ceil(len(self.img_frames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    # from shuffled indices get the indices which will make the next batch 
    inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
    
    batch_img_frame = []
    batch_sem_masks = []
    batch_inst_masks = []
    # loading data from those indices to arrays
    for i in inds:
        batch_img_frame.append(self.img_frames[i])
        batch_sem_masks.append(self.sem_masks[i])
        batch_inst_masks.append(self.inst_masks[i])
        
    
    train_image = []
    train_h = []
    train_label_seg = []
    train_label_inst = []
    train_label_clf = []
    names = []
    for i in range(0, len(batch_img_frame)):
      img_path = batch_img_frame[i]
      sem_mask = batch_sem_masks[i]
      inst_mask = batch_inst_masks[i]
      clf_label = np.zeros(self.tumor_types.shape)
      # get classifier label
      for idx in range(len(self.tumor_types)):
        if os.path.basename(img_path).find(self.tumor_types[idx]) == -1:
            pass
        else:
            clf_label[idx] = 1
      names.append(os.path.basename(img_path))
      image, semantic, instance, h_comp = Tumor_IO(img_path, sem_mask, inst_mask, self.modelip_img_w, self.modelip_img_h, self.data_augment)
      img_n_h_comp = np.concatenate((image, h_comp), axis=-1)
      train_image.append(img_n_h_comp)
      train_h.append(h_comp)
      train_label_seg.append(semantic)
      train_label_inst.append(instance)
      train_label_clf.append(clf_label)
      
      
    #   op = show_results((np.array(train_image)[0,::])*255, np.array(train_label)[0,::], classes_name, self.modelip_img_w, self.modelip_img_h)
    #   cv2.imwrite('/home/user01/data_ssd/Talha/data_gen_test/img_{}.jpg'.format(self.i+1), op)
    #   self.i = self.i+1
      outputs = (np.array(train_label_clf), np.array(train_label_seg), np.array(train_label_inst))
      
      
    return np.array(train_image), outputs

