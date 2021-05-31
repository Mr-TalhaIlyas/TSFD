import tensorflow as tf
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D, MinPooling2D
if int(str(tf.__version__)[0]) == 2:
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, DepthwiseConv2D
    from tensorflow.keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    import tensorflow.keras.backend as K
    from tensorflow.keras.applications import EfficientNetB4, EfficientNetB0
    from tensorflow.keras.layers.experimental.preprocessing import Resizing
    from tensorflow.keras.regularizers import L2
if int(str(tf.__version__)[0]) == 1:
    from keras.models import Model
    from keras.regularizers import l2, l1
    from keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, DepthwiseConv2D
    from keras.layers import MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, Dropout
    from keras.optimizers import Adam, Nadam, SGD
    import keras.backend as K

from bifpn import BiFPN, ReSize, Half_BiFPN, OP_BiFPN, Upsampling_block
import copy
def use_customdropout():
    use_mydropout = False     # 1 for True
    return use_mydropout  # 0 for False

use_mydropout = use_customdropout()

from layers import DropBlock2D

if use_mydropout == True:
        from layers import DropBlock2D, Dropout, SpatialDropout2D
elif use_mydropout == False:
    if int(str(tf.__version__)[0]) == 1:
        from keras.layers import Dropout, SpatialDropout2D
    if int(str(tf.__version__)[0]) == 2:
        from tensorflow.keras.layers import Dropout, SpatialDropout2D
        

'''
For binary segmentation set num_class=2
'''
def num_of_classes():
    num_class = 6
    return num_class

num_class = num_of_classes()
if num_class == 2:
    output_ch = 1
else:
    output_ch = num_class

use_mydropout = use_customdropout()

from layers import DropBlock2D

if use_mydropout == True:
        from layers import DropBlock2D, Dropout, SpatialDropout2D
elif use_mydropout == False:
    if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.0':
        from keras.layers import Dropout, SpatialDropout2D
    if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.3.0'  or tf.__version__ == '2.2.0-rc2':
        from tensorflow.keras.layers import Dropout, SpatialDropout2D
        
def DROPOUT(input_tensor, dropout_rate, Block_size, dropout_method = 'VD'):
    
    if dropout_method == 'DB':
        x = DropBlock2D(block_size=Block_size, rate=dropout_rate)(input_tensor)
    elif dropout_method == 'VD':
        x = Dropout(dropout_rate)(input_tensor)
    elif dropout_method == 'SD':
        x = SpatialDropout2D(dropout_rate)(input_tensor)
        
    return x

#%%
def tumor_hr(input_img, dropout=0.3, enc_dropout_type= 'SD', dec_dropout_type= 'SD', Block_size= 5):
    
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    # make layers name unique
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    ef4.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/ef4.h5', by_name=True)
    
    p2_4s = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3_4s = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    # p4_4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5_4s = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    # p6_4 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7_4s = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    c6 = PDC(p7_4s, 256)
    c6 = DROPOUT(c6, dropout, Block_size, dropout_method = dec_dropout_type)
    
    
    # Expanding Path
    dam6 = DAM(p5_4s, c6, 256, 15, 4, activation=tf.nn.swish)
    u6 = UpSampling2D(interpolation='bilinear')(dam6)

    dam7 = DAM(p3_4s, u6, 128, 15, 4, activation=tf.nn.swish)   
    u7 = UpSampling2D(interpolation='bilinear')(dam7)
    
    dam8 = DAM(p2_4s, u7, 64, 15, 8, activation=tf.nn.swish)   
    u8 = UpSampling2D(size=(2, 2), interpolation='bilinear')(dam8)
    
    seg_output = Conv2D(6, (5, 5), kernel_initializer = 'he_normal', padding = 'same', name='seg0')(u8)
    seg_output = Conv2D(6, (1, 1), kernel_initializer = 'he_normal', padding = 'same', name='seg_out')(seg_output)# , activation='softmax'
    '''
    inst Seg Model Start
    '''
    #connecting encoder decoder
    c6i = PDC(p7_4s, 256)
    c6i = DROPOUT(c6i, dropout, Block_size, dropout_method = dec_dropout_type)
    
    
    # Expanding Path
    dam6i = DAM(p5_4s, c6i, 256, 15, 4, activation=tf.nn.swish)
    u6i = UpSampling2D(interpolation='bilinear')(dam6i)

    dam7i = DAM(p3_4s, u6i, 128, 15, 4, activation=tf.nn.swish)   
    u7i = UpSampling2D(interpolation='bilinear')(dam7i)
    
    dam8i = DAM(p2_4s, u7i, 64, 15, 8, activation=tf.nn.swish)   
    u8i = UpSampling2D(size=(2, 2), interpolation='bilinear')(dam8i)
    
    inst_output = Conv2D(1, (5, 5), kernel_initializer = 'he_normal', padding = 'same', name='inst0')(u8i)
    inst_output = Conv2D(1, (1, 1), kernel_initializer = 'he_normal', padding = 'same', name='inst_out')(inst_output)
    
    model = Model(inputs=[input_img], outputs=[seg_output, inst_output])
    
    return model
#%%
    
def efficent_pet_2(input_img, output_ch, bifpn_ch = 224):
    
    '''
    Extract RGB features
    '''
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Input layers of builtin models have the rescaling and normalization layers
    # so dont noramlize or rescale the data yourself inside the data generator
    
    # change the names of 1st three layers so that we can uplaod pretrained weights of 
    # imagenet. As pretrined models take 3 channel input but we are giving it 6 channel
    # input so we will change the name of 1st few (6) layers before the MBConv1 block starts
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    ef4.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/ef4.h5', by_name=True)
        
    p2 = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3 = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    p4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5 = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    p6 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7 = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    '''
    Make Inputs for BiFPN
    '''
    
    # set channels W_bi_fpn to 128
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, bifpn_layer = 1)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, bifpn_layer = 2)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, bifpn_layer = 3)
    
    '''
    Seg Branch
    '''
    p2_seg = Half_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='seg')
    
    p2_seg = UpSampling2D(interpolation='bilinear')(p2_seg)# nearest
    seg_output = Conv2D(output_ch, (1, 1), padding='same', name='seg_out')(p2_seg)
    
    '''
    Inst Branch
    '''
    
    p2_inst = Half_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='inst')
    
    p2_inst = UpSampling2D(interpolation='bilinear')(p2_inst)# nearest
    inst_output = Conv2D(1, (1, 1), padding='same', name='inst_out')(p2_inst)
    
    model = Model(inputs=input_img, outputs=[seg_output, inst_output])
    
    return model

#%%
def efficent_pet_201(input_img, output_ch, bifpn_ch = 224, dropout_rate=0.3, use_dropout=False):
    
    '''
    Extract RGB features
    '''
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Input layers of builtin models have the rescaling and normalization layers
    # so dont noramlize or rescale the data yourself inside the data generator
    
    # change the names of 1st three layers so that we can uplaod pretrained weights of 
    # imagenet. As pretrined models take 3 channel input but we are giving it 6 channel
    # input so we will change the name of 1st few (6) layers before the MBConv1 block starts
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    ef4.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/ef4.h5', by_name=True)
        
    p2 = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3 = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    p4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5 = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    p6 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7 = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    '''
    Make Inputs for BiFPN
    '''
    
    # set channels W_bi_fpn to 128
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    if use_dropout:
        p7 = SpatialDropout2D(dropout_rate, name='drop_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    if use_dropout:
        p6 = SpatialDropout2D(dropout_rate, name='drop_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    if use_dropout:
        p5 = SpatialDropout2D(dropout_rate, name='drop_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    if use_dropout:
        p4 = SpatialDropout2D(dropout_rate, name='drop_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    if use_dropout:
        p3 = SpatialDropout2D(dropout_rate, name='drop_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    if use_dropout:
        p2 = SpatialDropout2D(dropout_rate, name='drop_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, 
                                               bifpn_layer = 1, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, 
                                               bifpn_layer = 2, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, 
                                               bifpn_layer = 3, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    '''
    Seg Branch
    '''
    p2_seg = Half_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='seg')
    
    p2_seg = UpSampling2D(interpolation='bilinear')(p2_seg)# nearest
    seg_output = Conv2D(output_ch, (1, 1), padding='same', name='seg_out')(p2_seg)
    
    '''
    Inst Branch
    '''
    
    p2_inst = Half_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='inst')
    
    p2_inst = UpSampling2D(interpolation='bilinear')(p2_inst)# nearest
    inst_output = Conv2D(1, (1, 1), padding='same', name='inst_out')(p2_inst)
    
    model = Model(inputs=input_img, outputs=[seg_output, inst_output])
    
    return model
#%%
def efficent_pet_203(input_img, output_ch, bifpn_ch = 224, dropout_rate=0.3, use_dropout=False):
    
    '''
    Extract RGB features
    '''
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Input layers of builtin models have the rescaling and normalization layers
    # so dont noramlize or rescale the data yourself inside the data generator
    
    # change the names of 1st three layers so that we can uplaod pretrained weights of 
    # imagenet. As pretrined models take 3 channel input but we are giving it 6 channel
    # input so we will change the name of 1st few (6) layers before the MBConv1 block starts
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    ef4.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/ef4.h5', by_name=True)
        
    p2 = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3 = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    p4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5 = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    p6 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7 = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    '''
    Make Inputs for BiFPN
    '''
    
    # set channels W_bi_fpn to 128
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    if use_dropout:
        p7 = SpatialDropout2D(dropout_rate, name='drop_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    if use_dropout:
        p6 = SpatialDropout2D(dropout_rate, name='drop_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    if use_dropout:
        p5 = SpatialDropout2D(dropout_rate, name='drop_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    if use_dropout:
        p4 = SpatialDropout2D(dropout_rate, name='drop_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    if use_dropout:
        p3 = SpatialDropout2D(dropout_rate, name='drop_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    if use_dropout:
        p2 = SpatialDropout2D(dropout_rate, name='drop_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, 
                                               bifpn_layer = 1, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, 
                                               bifpn_layer = 2, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, 
                                               bifpn_layer = 3, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    '''
    Top-Down FPNs for Seg and Inst branch
    '''
    p2_seg, p2_inst = OP_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='out')#, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    '''
    Seg Branch
    '''    
    p2_seg = Upsampling_block(p2_seg, output_ch)
    seg_output = DepthwiseConv2D((3, 3), padding='same', name='seg_out')(p2_seg)# a 3x3 conv followed by 1x1
    
    '''
    Inst Branch
    '''    
    p2_inst = Upsampling_block(p2_inst, 1)
    inst_output = DepthwiseConv2D((3, 3), padding='same', name='inst_out')(p2_inst)# a 3x3 conv followed by 1x1
    
    model = Model(inputs=input_img, outputs=[seg_output, inst_output])
    
    return model

#%%
def efficent_pet_203_clf(input_img, output_ch, bifpn_ch = 224, dropout_rate=0.3, use_dropout=False):
    
    '''
    Extract RGB features
    '''
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Input layers of builtin models have the rescaling and normalization layers
    # so dont noramlize or rescale the data yourself inside the data generator
    
    # change the names of 1st three layers so that we can uplaod pretrained weights of 
    # imagenet. As pretrined models take 3 channel input but we are giving it 6 channel
    # input so we will change the name of 1st few (6) layers before the MBConv1 block starts
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    ef4 = EfficientNetB4(include_top=False, weights=None, input_tensor=input_img)
    for i in range(6):
        ef4.layers[i]._name = ef4.layers[i].get_config()['name']+ '_pet'
        
    ef4.load_weights('/home/user01/data_ssd/Talha/pannuke/pan_idx/ef4.h5', by_name=True)
        
    p2 = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3 = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    p4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5 = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    p6 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7 = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    '''
    Make Inputs for BiFPN
    '''
    
    # set channels W_bi_fpn to 128
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    if use_dropout:
        p7 = SpatialDropout2D(dropout_rate, name='drop_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    if use_dropout:
        p6 = SpatialDropout2D(dropout_rate, name='drop_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    if use_dropout:
        p5 = SpatialDropout2D(dropout_rate, name='drop_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    if use_dropout:
        p4 = SpatialDropout2D(dropout_rate, name='drop_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    if use_dropout:
        p3 = SpatialDropout2D(dropout_rate, name='drop_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    if use_dropout:
        p2 = SpatialDropout2D(dropout_rate, name='drop_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, 
                                               bifpn_layer = 1, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, 
                                               bifpn_layer = 2, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, 
                                               bifpn_layer = 3, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    
    '''
    Top-Down FPNs for Seg and Inst branch
    '''
    p2_seg, p2_inst = OP_BiFPN(p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, bifpn_layer='out')#, Dropout_Rate=dropout_rate, Use_Dropout=use_dropout)
    '''
    Seg Branch
    '''    
    p2_seg = Upsampling_block(p2_seg, output_ch)
    seg_output = DepthwiseConv2D((3, 3), padding='same', name='seg_out')(p2_seg)# a 3x3 conv followed by 1x1
    
    '''
    Inst Branch
    '''    
    p2_inst = Upsampling_block(p2_inst, 1)
    inst_output = DepthwiseConv2D((3, 3), padding='same', name='inst_out')(p2_inst)# a 3x3 conv followed by 1x1
    
    '''
    Clf Branch
    '''  
    gap = GlobalAveragePooling2D(name='GAP_clf')(p7)
    FC1 = Dropout(0.3)(gap)
    clf_output = Dense(19, activation='softmax', name='clf_out')(FC1) 
    model = Model(inputs=input_img, outputs=[clf_output, seg_output, inst_output])
    return model
#%%
'''This is the 1st one I made with clf branch included 3 IOs'''
def efficent_pet_d4(rgb_img, h_img, output_ch, bifpn_ch = 224):
    '''This is the 1st one I made with clf branch included'''
    '''
    Extract RGB features
    '''
    ef4 = EfficientNetB4(include_top=False, weights="imagenet", input_shape=(1024, 1024, 3), input_tensor=rgb_img)
    # make layers name unique
    for layer in ef4.layers:
        layer._name = layer.get_config()['name']+ '_D4'
        
    p2_4 = ef4.layers[87].output # shape=(None, 256, 256, 32) {'name': 'block2d_add',
    p3_4 = ef4.layers[146].output # shape=(None, 128, 128, 56)  {'name': 'block3d_add'
    p4_4 = ef4.layers[235].output # shape=(None, 64, 64, 112) {'name': 'block4f_add'
    p5_4 = ef4.layers[323].output # shape=(None, 64, 64, 160) {'name': 'block5f_add',
    p6_4 = ef4.layers[442].output # shape=(None, 32, 32, 272) {'name': 'block6h_add',
    p7_4 = ef4.layers[473].output # shape=(None, 32, 32, 1792) {'name': 'top_activation',
    
    '''
    Extract Hematocyline features
    '''
    ef0 = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(1024, 1024, 3), input_tensor=h_img)
    # make layers name unique
    for layer in ef0.layers:
        layer._name = layer.get_config()['name']+ '_D0'
        
    p2_0 = ef0.layers[45].output # shape=(None, 256, 256, 24) {'name': 'block2b_add'
    p3_0 = ef0.layers[74].output # shape=(None, 128, 128, 40)  {'name': 'block3b_add'
    p4_0 = ef0.layers[118].output # shape=(None, 64, 64, 80) {'name': 'block4c_add', 
    p5_0 = ef0.layers[161].output # shape=(None, 64, 64, 112) {'name': 'block5c_add',
    p6_0 = ef0.layers[220].output # shape=(None, 32, 32, 192) {'name': 'block6d_add', '
    p7_0 = ef0.layers[236].output # shape=(None, 32, 32, 1280) {'name': 'top_activation'
    
    # concate both backbone features
    
    p2 = concatenate([p2_4, p2_0])
    p3 = concatenate([p3_4, p3_0])
    p4 = concatenate([p4_4, p4_0])
    p5 = concatenate([p5_4, p5_0])
    p6 = concatenate([p6_4, p6_0])
    p7 = concatenate([p7_4, p7_0])
    '''
    Make Inputs for BiFPN
    '''
    
    # set channels W_bi_fpn to 128
    p7 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p7in_bifpn')(p7)
    p7 = BatchNormalization(name='bn_p7in_bifpn')(p7)
    p7 = Activation(tf.nn.swish, name='act_p7in_bifpn')(p7)
    
    p6 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p6in_bifpn')(p6)
    p6 = BatchNormalization(name='bn_p6in_bifpn')(p6)
    p6 = Activation(tf.nn.swish, name='act_p6in_bifpn')(p6)
    
    p5 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p5in_bifpn')(p5)
    p5 = BatchNormalization(name='bn_p5in_bifpn')(p5)
    p5 = Activation(tf.nn.swish, name='act_p5in_bifpn')(p5)
    
    p4 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p4in_bifpn')(p4)
    p4 = BatchNormalization(name='bn_p4in_bifpn')(p4)
    p4 = Activation(tf.nn.swish, name='act_p4in_bifpn')(p4)
    
    p3 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p3in_bifpn')(p3)
    p3 = BatchNormalization(name='bn_p3in_bifpn')(p3)
    p3 = Activation(tf.nn.swish, name='act_p3in_bifpn')(p3)
    
    p2 = Conv2D(bifpn_ch, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(4e-5), name='conv_p2in_bifpn')(p2)
    p2 = BatchNormalization(name='bn_p2in_bifpn')(p2)
    p2 = Activation(tf.nn.swish, name='act_p2in_bifpn')(p2)
    '''
    Start BiFPN
    '''
    p2_1, p3_1, p4_1, p5_1, p6_1, p7_1 = BiFPN(p2, p3, p4, p5, p6, p7, bifpn_layer = 1)
    
    p2_2, p3_2, p4_2, p5_2, p6_2, p7_2 = BiFPN(p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, bifpn_layer = 2)
    
    p2_3, p3_3, p4_3, p5_3, p6_3, p7_3 = BiFPN(p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, bifpn_layer = 3)
    
    '''
    Clf branch
    '''
    p3_3 = Conv2D(bifpn_ch, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=L2(4e-5), name='conv_clf0')(p3_3)
    p3_3 = BatchNormalization(name='bn_clf0')(p3_3)
    p3_3 = Activation(tf.nn.swish, name='act_clf0')(p3_3)
    
    p345 = add([p4_3, p5_3, p3_3], name='add_clf_p345')
    p345 = Conv2D(bifpn_ch, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=L2(4e-5), name='conv_clf1')(p345)
    p345 = BatchNormalization(name='bn_clf1')(p345)
    p345 = Activation(tf.nn.swish, name='act_clf1')(p345)
    
    p_all = add([p6_3, p7_3, p345], name='add_clf_p34567')
    p_all = Conv2D(bifpn_ch, (3, 3), padding='same', kernel_regularizer=L2(4e-5), name='conv_clf2')(p_all)
    p_all = BatchNormalization(name='bn_clf2')(p_all)
    p_all = Activation(tf.nn.swish, name='act_clf2')(p_all)
    
    p_all = Conv2D(1792, (3, 3), padding='same', kernel_regularizer=L2(4e-5), name='conv_clf3')(p_all)
    p_all = BatchNormalization(name='bn_clf3')(p_all)
    p_all = Activation(tf.nn.swish, name='act_clf3')(p_all)
    
    gap = GlobalAveragePooling2D(name='GAP_clf')(p_all)
    FC1 = Dropout(0.3)(gap)
    clf_output = Dense(19, activation='softmax', name='clf_out')(FC1) 
    '''
    Seg Branch
    '''
    
    seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_seg_1')(p2_3)
    seg = BatchNormalization(name='bn_seg_1')(seg)
    seg = Activation(tf.nn.swish, name='act_seg_1')(seg)
    
    seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_seg_2')(seg)
    seg = BatchNormalization(name='bn_seg_2')(seg)
    seg = Activation(tf.nn.swish, name='act_seg_2')(seg)
    
    seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_seg_3')(seg)
    seg = BatchNormalization(name='bn_seg_3')(seg)
    seg = Activation(tf.nn.swish, name='act_seg_3')(seg)
    
    seg = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_seg_4')(seg)
    seg = BatchNormalization(name='bn_seg_4')(seg)
    seg = Activation(tf.nn.swish, name='act_seg_4')(seg)
    
    seg_output = Conv2D(output_ch, (1, 1), padding='same', name='seg_out')(seg)
    
    '''
    Inst Branch
    '''
    
    inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_inst_1')(p2_3)
    inst = BatchNormalization(name='bn_inst_1')(inst)
    inst = Activation(tf.nn.swish, name='act_inst_1')(inst)
    
    inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_instg_2')(inst)
    inst = BatchNormalization(name='bn_inst_2')(seg)
    inst = Activation(tf.nn.swish, name='act_inst_2')(seg)
    
    inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_inst_3')(inst)
    inst = BatchNormalization(name='bn_inst_3')(seg)
    inst = Activation(tf.nn.swish, name='act_inst_3')(seg)
    
    inst = DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=L2(4e-5), name='dw_inst_4')(inst)
    inst = BatchNormalization(name='bn_inst_4')(inst)
    inst = Activation(tf.nn.swish, name='act_inst_4')(inst)
    
    inst_output = Conv2D(1, (1, 1), padding='same', name='inst_out')(p2)
    
    model = Model(inputs=[rgb_img, h_img], outputs=[clf_output, seg_output, inst_output])
    
    return model

#%%
# ef0 = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(1024, 1024, 3))
# x = [44, 73, 117, 160, 219, 232, 235]  

# for i in x:
#     print(i+1, ef0.layers[i+1].output)
#     print(i+1, ef0.layers[i+1].get_config())
#     print('\n')

# ef4 = EfficientNetB4(include_top=False, weights="imagenet", input_shape=(1024, 1024, 3))
# x = [87,146,235,323,442,473]  

# for i in x:
#     print(i, ef4.layers[i].output)
#     print(i, ef4.layers[i].get_config())
#     print('\n')