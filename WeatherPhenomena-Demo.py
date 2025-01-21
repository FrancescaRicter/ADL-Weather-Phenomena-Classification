import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder

import requests
import numpy as np
import os

IMG_ROWS = 224
IMG_COLS = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Model tuned hyper-parameters
ACTIVATION_FUNCTION = "swish"
LEARNING_RATE = 0.001

MODEL_URL = "https://github.com/FrancescaRicter/ADL-Weather-Phenomena-Classification/releases/download/v1.1.0/checkpoint.weights.h5"  
MODEL_PATH = "checkpoint.weights.h5"

# Model architecture 
def mbconv_block(
    x,
    filters,
    activation_function,
    expansion_ratio=4,
    kernel_size=3,
    stride=1,
    use_se=True,):
    """
    MobileNetV2's Mobile Inverted Residual Block with depthwise convolution, optional Squeeze-and-Excitation (SE).

    Args:
        x (tf.Tensor): Input tensor to the block (with shape (batch_size, height, width, in_channels)).
        filters (int): The number of output filters (channels) for the final convolution.
        activation_function (string)
        expansion_ratio (int): The factor by which to expand the number of channels in the expansion phase
        kernel_size (int): The size of the depthwise convolution kernel
        stride (int): The stride of the depthwise convolution
        use_se (bool): Whether to use the Squeeze-and-Excitation (SE) block

    Returns:
        x (tf.Tensor) with shape (batch_size, height, width, filters), where `filters` is the number of output channels passed as an argument to the function.
    """
    in_channels = x.shape[-1]
    expanded_channels = in_channels * expansion_ratio

    # Expansion phase
    if expansion_ratio > 1:
        x = layers.Conv2D(
            expanded_channels, kernel_size=1, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_function)(x)

    # Depthwise convolution phase
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)

    # Squeeze-and-Excitation phase
    if use_se:
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(in_channels // 16, activation=activation_function)(se)
        se = layers.Dense(expanded_channels, activation="sigmoid")(se)
        se = layers.Reshape((1, 1, expanded_channels))(se)
        x = layers.Multiply()([x, se])

    # Apply 1x1 convolution
    x = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    return x

def model_EfficientNetB0(input_shape, num_classes, activation_function):
    """
    This method builds the EfficientNetB0 model
    Args:
        input_shape (tuple): The shape of the input tensor, typically of the form (height, width, channels).
        num_classes (int): The number of output classes for classification. This defines the size of the final output layer.
        activation_function (string): The activation function to be used throughout the model, typically 'relu', 'swish' etc..

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # Apply Conv 3x3 with stride 2
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)

    configs = [
        # Configurations for the various MBConv Blocks
        # (filters, expansion_ratio, kernel_size, strides, repeats)
        (16, 1, 3, 1, 1),
        (24, 6, 3, 2, 2),
        (40, 6, 5, 2, 2),
        (80, 6, 3, 2, 3),
        (112, 6, 5, 1, 3),
        (192, 6, 5, 2, 4),
        (320, 6, 3, 1, 1),
    ]

    # Create all the mbconv_blocks by iterating thrugh the various configurations
    for filters, expansion_ratio, kernel_size, stride, repeats in configs:
        for i in range(repeats):
            stride = stride if i == 0 else 1
            x = mbconv_block(
                x, filters, activation_function, expansion_ratio, kernel_size, stride
            )

    # Apply a 1x1 convolution
    x = layers.Conv2D(1280, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

# download the model weights from github release
def download_model(url, output_path):
    if not os.path.exists(output_path):
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

# load LabelEncoder classes
@st.cache_resource
def load_label_encoder():
    le = LabelEncoder()
    try:
        le.classes_ = np.load("classes.npy")
    except Exception as e:
        st.error(f"Failed to load class labels: {e}")
        st.stop()
    return le

# load model
@st.cache_resource
def load_trained_model():
    download_model(MODEL_URL, MODEL_PATH)

    model = model_EfficientNetB0(INPUT_SHAPE, len(label_encoder.classes_), ACTIVATION_FUNCTION)

    OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=OPTIMIZER,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
      
    # load the weights of the model
    try:
        model.load_weights(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()
    return model

# Preprocess the image
def preprocess_image(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        image = tf.image.decode_image(file_bytes, channels=3)
        image = tf.image.resize(image, [IMG_ROWS, IMG_COLS])
        image = image / 255.0
    except Exception as e:
        st.error(f"Failed to process image: {e}")
        st.stop()
    return tf.expand_dims(image, axis=0)

st.title("Weather Phenomena Image Classifier")
st.write("Upload an image to classify the represented weather phenomenon.")

# file upload
uploaded_file = st.file_uploader("Choose a JPG or JPEG image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # load resources
    label_encoder = load_label_encoder()
    model = load_trained_model()

    image_tensor = preprocess_image(uploaded_file)

    # predict only when a new file is uploaded and the button is clicked by the user
    if st.button("Classify Image"):
        y_pred = model.predict(image_tensor, verbose=0)
        y_pred_transformed = np.argmax(y_pred[0])
        categorical_label = label_encoder.inverse_transform([y_pred_transformed])

        st.write(f"The image represents: **{categorical_label[0]}**")
