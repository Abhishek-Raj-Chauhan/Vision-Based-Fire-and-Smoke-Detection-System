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

![20BCI0161_VL2023240104782_PE00kkkk3](https://github.com/Abhishek-Raj-Chauhan/Vision-Based-Fire-and-Smoke-Detection-System/assets/100334669/2a73c204-398c-4dec-9a35-f783a168944e)

The proposed architecture for fire and smoke detection leverages the Inception V3 convolutional neural network (CNN), a robust and efficient architecture renowned for image recognition tasks. This architecture is divided into two primary components: the convolutional base and the classification head. Inception-v3 brings several notable enhancements to the table, including the adoption of Label Smoothing, Factorized 7 x 7 convolutions, and the incorporation of an auxiliary classifier to disseminate label information deeper into the network. Additionally, batch normalization is employed for layers in the side head, contributing to the overall effectiveness of the architecture.

Convolutional Base: The convolutional base of Inception V3 consists of multiple layers that extract relevant features from input images. It includes convolutional layers arranged hierarchically to learn features at different levels of abstraction. Lower-level layers capture basic features like edges and corners, while higher- level layers identify complex patterns and textures. Pooling layers are used to downsample feature maps and reduce spatial dimensions.

Classification Head: The classification head follows the convolutional base and is responsible for the final classification of images. It starts with a global average pooling layer to compute the average of feature maps across spatial dimensions. A dense layer follows, performing the actual classification. 

Rectified Linear Unit (ReLU) activation introduces non-linearity to capture complex relationships. Softmax activation produces a probability distribution over classes, determining the class with the highest probability.

ReLU Activation: Rectified Linear Unit (RELU) is a very effective and simple activation function; it acts as a nonlinear activation function and linear activation function. As it returns the value provided as input without any transformation or returns 0 if the input value is 0 or less. It computationally defined as equation.

![20BCI0161_VL2023240104782_PE00kkkk3](https://github.com/Abhishek-Raj-Chauhan/Vision-Based-Fire-and-Smoke-Detection-System/assets/100334669/93e51158-8e2c-40dd-ba54-26d07179cf0f)

Mean Pooling Layer: Inception-v3 does not include a specific "Mean Pooling Layer" in the traditional sense. However, it does utilize global average pooling (GAP) as part of its architecture. After the convolutional layers, the GAP layer is applied to compute the average of each feature map, resulting in a single value for each feature map. This reduces the spatial dimensions and provides a more compact representation of the features. GAP is often used to replace the fully connected layers at the end of the network, reducing the risk of overfitting and making the network more robust.

Max Pooling Layer: Max pooling layers are used in Inception-v3 as part of the convolutional layers. Max pooling helps down sample the feature maps, reducing computational complexity and retaining the most important information from each region. It is applied at various stages of the network to capture different levels of abstraction.

Fully Connected Layer: Inception-v3 typically uses fully connected layers towards the end of the network. These fully connected layers are typically followed by a softmax layer to produce class probabilities. The fully connected layers help in learning complex relationships in the feature maps and are crucial for the final classification step.

Softmax Layer: The softmax layer is used at the end of the Inception-v3 network to produce probability scores for different classes. It takes the output of the fully connected layers and converts them into a probability distribution over the possible classes. The class with the highest probability is considered the final prediction of the network.

Dropout Layers: Dropout layers are a crucial tool in training neural networks, especially in deep learning, to address the problem of overfitting. Overfitting happens when a model becomes too specialized in learning the training data, which can lead to poor performance on unseen data. Dropout works by randomly deactivating a portion of neurons or units within a layer during training, thereby introducing a form of noise and redundancy in the network. This dropout process encourages the model to learn more robust and generalized representations of the data, as it can't rely too heavily on any specific neurons. It essentially prevents the network from memorizing the training data and instead promotes a broader understanding of the underlying patterns, improving the model's ability to generalize to new, unseen examples.
