# Image Classification of Weather Phenomena
# Assignment 1 - Initiate

## How To Run the Weather Phenomena Demo
1. Prerequisites
    Ensure you have the following installed:
    - Python (>=3.7)
    - TensorFlow (2.x)
    - NumPy
    - scikit-learn

2. Required Files
    Ensure you have the following files in your working directory:
    - dataset.zip : Contains a sample of the data images from the final 13 different classes
    - classes.npy : Encoding of the class labels
    - WeatherPhenomena/trial_2/checkpoint.weights.h5 : Pre-trained model weights.

3. Run the following command in terminal:
   python WeatherPhenomenaDemo.py 

4. When the application requestes it, provide the path to the image you want to get predictions for.

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
The task remains the same, however the chosen architecture **VGG16** was cahnged as it is relatively old and better models have been developed with better reported accuracy. In particular the new architecture chosen for the project is **EfficientNetB0** this choice was driven by the high accuracy with fewer parameters and computational cost compared to other models.

## Working Pipeline 
1. Re-Implement Baseline model following the EfficientNetB0 architecture, decide metric and scores for a successful outcome. Implement unit tests. Run on the current WEAPD dataset as it is.
2. Gather more data and do further research into the topic for improvements.
3. Data Augmentation, Data preprocessing improve the model and fine tune hyper-parameters. Implement unit tests.
4. Final Evaluation of the model on the test data. Go back to point 3 if the desired outcome has not been achieved.

## Evaluation Metric
For evaluating the model, the accuracy performance of the model will be observed. The best accuracy reported for **EfficientNetB0** is 77.1%, while the best accuracy reported by the two research papers using the WEAPD dataset achieve a 92% accuracy outcome. Therefore, a successful outcome of this project would report an accuracy between 85%-95% on the test split of the data. 
For each epoch the performance for accuracy and loss will be observed for train and validation set.

## Work Summary
The **EfficientNetB0** was chosen for it's reduced number of parameters compared to other image classification models. 
The model built showed a total of 3.5M trainable parameters, comapred to the actual reported 5.3M parameters.
Two new calsses were introduced 'Tornado' and 'Aurora' and samples were downloaded from Bing using the 'bing_image_downloader' library.
Data Augmentation techniques, flipping , adjusting contrast or brightness of a small percentage of the data in the train-set. 
Oversampling was employed to overcome class imbalance. Finally, the total samples used for training increased from a total of 6862 to a total of 4899 to a total of 15387.
Tuning was performed for three hyper-parameters : activation_function, learning_rate and the optimizer. The best hyper parameters resulted being the same as the ones reported in scientific literature (activation_function = 'swish', learning_rate = '0.001', optimizer = 'Root-mean squared propagation (RMSProp)')

## Results 
The model uses the RMSProp optimizer, a desireble outcome would be to achieve a loss below 0.4 for the test set.

The model was trained for a total of 32 epochs. The accuracy performances reported are:
- Train set : 93%
- Validation set : 87%
- Test set : 91% (Loss: 0.27)
Based on these results, we can confirm that the success criteria has been met.

## Observations
During test-validation split an error was performed when setting the size for split percentage to 0.15 (15%), leading to the following split for the original WEAPD dataset : 68% train, 12% validation and 20% test. This error should be fixed to have more equally balanced number of validation and test data samples.

## Work Breakdown
- **Dataset Retrieval and Scientific Paper Research:** 3 hours
- **Data Preprocessing, Model Development, and Training:** 17.45 hours
- **Model Improvements and Fine-Tuning:** 16.10 hours
- **Final Evaluation:** 0.30 hours
- **Report/READ ME and Results Presentation:** 1.15 hours 
