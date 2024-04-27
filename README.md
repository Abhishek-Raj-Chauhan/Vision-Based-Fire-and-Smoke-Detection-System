# Vision-Based Fire and Smoke Detection System
 The aim of this project is real-time and accurate identification of fire and smoke occurrences to enhance disaster management and emergency response strategies.Utilized Convolutional Neural Networks (CNNs) with InceptionV3 and AlexNet architectures, along with data preprocessing, data augmentation, multiclass classification with softmax, ReLU activation, evaluation metrics such as accuracy, and frameworks like Keras and TensorFlow for efficient model development and deployment.

<img width="929" alt="Screenshot 2024-04-27 212759" src="https://github.com/Abhishek-Raj-Chauhan/Vision-Based-Fire-and-Smoke-Detection-System/assets/100334669/5dc952fe-d091-4aa2-98fe-d53fdacfdecc">

# Proposed Methodology:

Dataset collection is an essential process that involves gathering relevant data to train a model for a specific task. In this process, various techniques are used to collect data from various sources, online
databases, web scraping, social media platforms, and many others. One of the critical steps in the dataset collection process is data scrubbing or cleaning. Data scrubbing involves the process of detecting and correcting or removing inaccurate, incomplete, or irrelevant data from a dataset. This process helps us to ensure that the dataset used for training is accurate, consistent, and of high quality. In the case of fire and smoke detection, dataset collection involves gathering data on fire and smoke using various sources. The dataset we collected include images of fire and smoke, as well as non-fire and smoke images around 900 each for training the model and for testing and validation we have 100 images each.

The collected dataset then scrubbed to remove any irrelevant or incorrect data, ensuring that the dataset is clean and suitable for training a machine learning model.
During the data scrubbing process, several techniques are used to identify and clean the dataset. These techniques include data profiling, data standardization, data transformation, and data enrichment. Data profiling involves analysing the dataset to identify any inconsistencies, duplicates, or missing data. Data standardization involves converting data to a common format, such as converting dates to a standard format. Data transformation involves converting data into a form suitable for analysis, such as
converting text data into numerical data. Data enrichment involves adding additional data to the dataset
to improve its quality and completeness.

Once the dataset has been cleaned, we used it to train machine learning model to detect fire and smoke.
Machine learning algorithm such as convolutional neural networks (CNNs) are commonly used for
image recognition tasks, such as fire and smoke detection. It is trained using the cleaned dataset to identify patterns and features that distinguish fire and smoke from non-fire and smoke images.
Validation and testing sets have been combined into one. Our dataset largely consists of three classes.
Fire, Neutral, and Smoke are them. Each class has approximately 100 images for testing and over 900 images for training.

# Architectural Framework

The proposed architecture for fire and smoke detection leverages the Inception V3 convolutional neural network (CNN), a robust and efficient architecture renowned for image recognition tasks. This architecture is divided into two primary components: the convolutional base and the classification head. Inception-v3 brings several notable enhancements to the table, including the adoption of Label Smoothing, Factorized 7 x 7 convolutions, and the incorporation of an auxiliary classifier to disseminate label information deeper into the network. Additionally, batch normalization is employed for layers in the side head, contributing to the overall effectiveness of the architecture.

Convolutional Base: The convolutional base of Inception V3 consists of multiple layers that extract relevant features from input images. It includes convolutional layers arranged hierarchically to learn features at different levels of abstraction. Lower-level layers capture basic features like edges and corners, while higher- level layers identify complex patterns and textures. Pooling layers are used to downsample feature maps and reduce spatial dimensions.

Classification Head: The classification head follows the convolutional base and is responsible for the final classification of images. It starts with a global average pooling layer to compute the average of feature maps across spatial dimensions. A dense layer follows, performing the actual classification. 

Rectified Linear Unit (ReLU) activation introduces non-linearity to capture complex relationships. Softmax activation produces a probability distribution over classes, determining the class with the highest probability.

ReLU Activation: Rectified Linear Unit (RELU) is a very effective and simple activation function; it acts as a nonlinear activation function and linear activation function. As it returns the value provided as input without any transformation or returns 0 if the input value is 0 or less. It computationally defined as equation.
