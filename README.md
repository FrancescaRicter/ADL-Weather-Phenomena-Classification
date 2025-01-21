# Image Classification of Weather Phenomena

## How To Run the Weather Phenomena Demo Web Application
1. Prerequisites
    Ensure you have the following installed:
    - Python (>=3.7)
    - TensorFlow (2.x)
    - NumPy
    - scikit-learn

2. Required Streamlit library, to install run the following command:
    *pip install streamlit*

3. Run the following command in terminal to access the WebApp :
   streamlit run WeatherPhenomena-Demo.py 

4. Upload your image file and perform inference

# Assignment 1 - Initiate

## For detailed information about the experiment, please refer to the Jupyter Notebook.

## 1. Introduction
In the persistently evolving field of Machine Learning, Computer Vision has gained enormous popularity due to its ability to interpret visual information meaningfully. This project proposes an Image Classification task using Deep Learning methods on the Weather Phenomena Dataset (WEAPD) to classify diverse weather phenomena.

## 2. Dataset
The dataset comprises **6862 photos** across **11 classes**: 
- Dew
- Fog/Smog
- Frost
- Glaze
- Hail
- Lightning
- Rain
- Rainbow
- Rime
- Sandstorm
- Snow

This dataset includes a higher number of classes compared to traditional weather classification datasets. The class with the most samples is **'rime'** with **1160 images**, while **'rainbow'** accounts for only **238 images**, making this an imbalanced dataset. To address this shortcoming, techniques such as data augmentation will be adopted. All samples are color images in JPG format that vary in size.

## 3. Related Works
In image classification tasks, transfer learning is commonly applied, particularly useful for addressing insufficient training data. Some of the best results on this dataset leverage the **VGG16** architecture.

The authors of the WEAPD dataset discuss their approach for Weather Classification in their 2021 paper, *"Classification of Weather Phenomenon From Images by Using Deep Convolutional Neural Network."* They introduce the **MeteCNN** architecture, a modified version of VGG16, achieving impressive performance with **92.68% accuracy**. This architecture includes improvements like the use of **Squeeze-and-Excitation (SE)** modules and dilated convolutions. A more recent paper published in 2024, *"Enhanced Multi-Class Weather Image Classification Using Data Augmentation and Dynamic Learning on Pre-Trained Deep Neural Net,"* utilizes transfer learning to achieve comparable results with **92.10% accuracy** through data augmentation techniques and fine-tuning.

## 4. Project Idea (Bring Your Own Method)
This project falls into the 'Bring Your Own Method' category for this assignment. It uses a pre-existing dataset and model, aiming to modify its architecture to achieve performance improvements. 

### Objectives:
1. **Expand the number of weather phenomena classes** by gathering additional data samples, as suggested by the dataset’s authors for future works.
2. **Explore additional data augmentation techniques** to address the data shortage for some classes.
3. **Re-implement the MeteCNN or VGG-16 models**, experimenting with modifications in the architecture, or fine-tuning the last layers (this will be assessed based on the computational feasibility of the training task), changing parameters such as learning rate, batch size, dropout, and experimenting with the loss function.

## 5. Work Breakdown
The development pipeline consists of the following steps:
- **Dataset Retrieval and Scientific Paper Research:** 3-4 hours
- **Data Preprocessing, Model Development, and Training:** 15-20 hours
- **Model Improvements and Fine-Tuning:** 15-20 hours
- **Final Testing and Evaluation:** 1-2 hours
- **Report and Results Presentation:** 6-8 hours

## References
1. H. Xiao, “Weather phenomenon database (WEAPD),” 2021. [Online]. Available: [https://doi.org/10.7910/DVN/M8JQCR](https://doi.org/10.7910/DVN/M8JQCR)
2. H. Xiao, F. Zhang, Z. Shen, K. Wu, and J. Zhang, “Classification of weather phenomenon from images by using deep convolutional neural network,” Earth and Space Science, vol. 8, 05 2021. [Online]. Available: (https://doi.org/10.1029/2020EA001604)
3. D. Chattoraj, A. Chatterjee, S. Ghosh, A. Ghosh, and E. Ientilucci, “Enhanced multi-class weather image classification using data augmentation and dynamic learning on pre-trained deep neural net,” 08 2024, pp. 459–464. [Online]. Available: (https://ieeexplore.ieee.org/document/10704109)


# Assignment 2 - Hacking

## Changes in the project 
The task remains the same, however the chosen architecture **VGG16** was changed as it is relatively old and better models have been developed with better reported accuracy. In particular the new architecture chosen for the project is **EfficientNetB0** this choice was driven by the high accuracy with fewer parameters and computational cost compared to other models.

## Evaluation Metric
For evaluating the model, the accuracy performance of the model will be observed. The best accuracy reported for **EfficientNetB0** is 77.1%, while the best accuracy reported by the two research papers using the WEAPD dataset achieve a 92% accuracy outcome. Therefore, a successful outcome of this project would report an accuracy between 85%-95% on the test split of the data. 
For each epoch the performance for accuracy and loss will be observed for train and validation set.

## Work Summary
The **EfficientNetB0** was chosen for its reduced number of parameters compared to other image classification models. 
The model built showed a total of 3.5M trainable parameters, compared to the actual reported 5.3M parameters.
Two new calsses were introduced 'Tornado' and 'Aurora' and samples were downloaded from Bing using the 'bing_image_downloader' library.
Data Augmentation techniques, such as flipping , adjusting contrast or brightness were applied on a small percentage of the data in the training set. 
Oversampling was employed to overcome class imbalance. Finally, the total samples used for training increased from a total of 4899 to a total of 15387.
Tuning was performed for three hyper-parameters : activation_function, learning_rate and the optimizer. The best hyper parameters resulted being the same as the ones reported in scientific literature (activation_function = 'swish', learning_rate = '0.001', optimizer = 'Root-mean squared propagation (RMSProp)')

## Results 
The model uses the RMSProp optimizer, a desireble outcome would be to achieve a loss below 0.4 for the test set.

The model was trained for a total of 32 epochs. The accuracy performances reported are:
- Train set : 93%
- Validation set : 87%
- Test set : 91% (Loss: 0.27)
Based on these results, we can confirm that the success criteria has been met.

## Observations
During the test-validation split, an error occurred when setting the split percentage to 15% (0.15). This resulted in the following distribution for the original WEAPD dataset: 68% for training, 12% for validation, and 20% for testing. This error caused an imbalance in the number of validation and test samples and could potentially explain the slightly lower performance on the validation data. In future work the split should be adjusted to ensure a more balanced distribution between validation and test data.

## Work Breakdown
- **Dataset Retrieval and Scientific Paper Research:** 3 hours
- **Data Preprocessing, Model Development, and Training:** 17.45 hours
- **Model Improvements and Fine-Tuning:** 16.10 hours
- **Final Evaluation:** 0.30 hours
- **Report/READ ME and Results Presentation:** 6.30 hours 

# Assignment 3 - Deliver
For this part of the project refer to the *Assignment3_Deliver.pdf* file.