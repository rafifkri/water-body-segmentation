"""
Model architectures for semantic segmentation
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE, IMG_CHANNELS


def unet_model(input_size=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name='unet'):
    """
    Build U-Net model for semantic segmentation
    
    Args:
        input_size: Input shape (height, width, channels)
        name: Model name
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_size)
    
    # Encoder
    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Block 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder
    # Block 6
    up6 = layers.UpSampling2D((2, 2))(conv5)
    up6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.UpSampling2D((2, 2))(conv6)
    up7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.UpSampling2D((2, 2))(conv7)
    up8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.UpSampling2D((2, 2))(conv8)
    up9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def unet_light(input_size=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name='unet_light'):
    """
    Lighter U-Net model (fewer parameters, faster training)
    
    Args:
        input_size: Input shape (height, width, channels)
        name: Model name
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_size)
    
    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Bottleneck
    bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(bottleneck)
    
    # Decoder
    up1 = layers.UpSampling2D((2, 2))(bottleneck)
    up1 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    merge1 = layers.concatenate([up1, conv3], axis=3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = layers.UpSampling2D((2, 2))(conv4)
    up2 = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
    merge2 = layers.concatenate([up2, conv2], axis=3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    up3 = layers.UpSampling2D((2, 2))(conv5)
    up3 = layers.Conv2D(32, 3, activation='relu', padding='same')(up3)
    merge3 = layers.concatenate([up3, conv1], axis=3)
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge3)
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv6)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def segnet_model(input_size=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name='segnet'):
    """
    SegNet-like architecture for semantic segmentation
    
    Args:
        input_size: Input shape (height, width, channels)
        name: Model name
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_size)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1, mask1 = layers.MaxPooling2D((2, 2), return_indices=True)(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2, mask2 = layers.MaxPooling2D((2, 2), return_indices=True)(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3, mask3 = layers.MaxPooling2D((2, 2), return_indices=True)(conv3)
    
    # Bottleneck
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(bottleneck)
    
    # Decoder
    unpool1 = layers.UpSampling2D((2, 2))(bottleneck)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(unpool1)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    
    unpool2 = layers.UpSampling2D((2, 2))(conv4)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(unpool2)
    conv5 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv5)
    
    unpool3 = layers.UpSampling2D((2, 2))(conv5)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(unpool3)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv6)
    
    model = keras.Model(inputs, outputs, name=name)
    return model


def get_model(model_type='unet', input_size=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)):
    """
    Get model based on type
    
    Args:
        model_type: Type of model ('unet', 'unet_light', 'segnet')
        input_size: Input shape
    
    Returns:
        Keras model
    """
    if model_type == 'unet':
        return unet_model(input_size)
    elif model_type == 'unet_light':
        return unet_light(input_size)
    elif model_type == 'segnet':
        return segnet_model(input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_model_summary(model):
    """Print model summary"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    print("="*60 + "\n")
