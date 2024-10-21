# Image Classification of Weather Phenomena

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
2. H. Xiao, F. Zhang, Z. Shen, K. Wu, and J. Zhang, “Classification of weather phenomenon from images by using deep convolutional neural network,” Earth and Space Science, vol. 8, 05 2021.
3. D. Chattoraj, A. Chatterjee, S. Ghosh, A. Ghosh, and E. Ientilucci, “Enhanced multi-class weather image classification using data augmentation and dynamic learning on pre-trained deep neural net,” 08 2024, pp. 459–464.
