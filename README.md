# German Traffic Sign Classification (GTS Classification)

This project focuses on classifying German traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It leverages a dataset of over **43,000** images encompassing **43** different traffic sign categories.  

**Project Highlights:**

* **Image Preprocessing:**
    - Employs `ImageDataGenerator` for data augmentation, including rescaling, shearing, zooming, horizontal flipping and brightness adjustment.
    - Images are resized to **64x64** pixels to ensure computational efficiency.
* **CNN Architecture:**
    - A custom CNN model is constructed, featuring multiple convolutional layers, max-pooling layers, and dropout layers for regularization.
    - The architecture incorporates `relu` activation for non-linearity.
    - The final output layer uses `softmax` activation to produce probabilities for each traffic sign class.
* **Training:**
    - The model is trained with the `adam` optimizer and utilizes categorical cross-entropy loss to measure performance.
    - Early stopping is implemented to prevent overfitting, monitoring accuracy with a `min_delta` of **0.01** and a `patience` of **2** epochs.
* **Evaluation:**
    - The trained model achieves a test accuracy of approximately **94.72%**.
* **Testing:**
    - The project provides functionality to test the model on individual images as well as a folder of test images.
    - Predictions are made by identifying the class with the highest probability score.




**Visualizations:**

**Screenshot 1: Model Architecture**
![Model Architecture](path/to/architecture_screenshot.png)

**Screenshot 2: Training Performance (Accuracy/Loss)**
![Training Performance](path/to/training_plot.png)

**Screenshot 3: Prediction on a Single Image**
![Single Image Prediction](path/to/single_image_prediction.png)

**Screenshot 4: Prediction on Multiple Images**
![Multiple Image Predictions]
(Images/Img_Pred_1.PNG)
(Images/Img_pred_2.PNG)



**Dependencies:**
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib


**Dataset:**
The project utilizes the GTSRB (German Traffic Sign Recognition Benchmark) dataset, which can be accessed through Kaggle.

**How to Run:**
1. Download the GTSRB dataset.
2. Install the necessary dependencies.
3. Execute the notebook in Google Colab to train and evaluate the model. 
4. Utilize the provided functions to predict on new traffic sign images.


**Acknowledgements:**
This project draws inspiration and knowledge from TensorFlow and Keras documentation, online tutorials, and publicly available resources on traffic sign classification.
