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

![20BCI0161_VL2023240104782_PE00kkkk3](https://github.com/Abhishek-Raj-Chauhan/Vision-Based-Fire-and-Smoke-Detection-System/assets/100334669/7701b98b-4473-4a95-8f02-90a0eb5390b2)

The final system integrates data from multiple sensors, in conjunction with surveillance system data, for comprehensive fire detection. By analyzing this combined information, the system determines whether a fire is present or not. This approach effectively reduces the occurrence of false positives and false negatives that are common in conventional methods.
Furthermore, the system employs a Dilated CNN (Convolutional Neural Network) to streamline the process, eliminating the need for labor-intensive handcrafted feature engineering. Instead, the Dilated CNN automatically extracts relevant and practical features to train the model, enhancing efficiency and accuracy in fire detection.

# Implementation

Convolutional neural networks (CNNs) are widely used in image recognition projects because they are
highly effective at detecting and recognizing visual patterns in images. CNNs are designed to automatically learn and extract features from input images, allowing them to identify complex patterns and relationships that are difficult or impossible for humans to recognize. CNNs consist of multiple layers of interconnected neurons that process image data in a hierarchical manner.

The Inception V3 architecture is a type of convolutional neural network that is commonly used for
image recognition tasks. It consists of 42 layers, including 4 inception modules that allow the network to capture multi-scale features at different levels of abstraction. The architecture consists of two main parts: the convolutional base and the classification head. The convolutional base is responsible for learning relevant features from the images, while the classification head is responsible for classifying
the images based on these features. The convolutional base of the Inception V3 architecture consists of several convolutional layers that learn relevant features from the images. The layers are arranged in a hierarchical fashion, with lower level layers learning basic features such as edges and corners, while higher level layers learn more complex features such as textures and patterns. The convolutional base also includes pooling layers that down sample the feature maps and reduce the spatial dimensions of the input images.

The classification head of the Inception V3 architecture consists of several fully connected layers that perform the final classification of the images. The classification head starts with a global average
pooling layer that computes the average of the feature maps across the spatial dimensions. The output of the global average pooling layer is then fed into a dense layer that performs the final classification.
The dense layer is followed by a ReLU activation function that introduces non-linearity into the model. Finally, the softmax activation function is used to produce a probability distribution over the output classes.

The next step we trained the Inception V3 model using the training dataset. During training, the model learns to extract relevant features from the images and classify them into fire, smoke, or non-fire and smoke categories. The model's performance during training is evaluated using the validation dataset.
The training process continues until the model reaches a satisfactory level of accuracy and
generalization. Keras and TensorFlow are two popular Python libraries used for image processing and deep learning. Keras is a high-level neural network API that runs on top of TensorFlow, making it easier to build and train deep learning models. TensorFlow is an open-source software library for dataflow and differentiable programming across a range of tasks, including machine learning and deep learning.

# Upload File or Real Time Detection

In this stage we use the well-trained Inception V3 model to classify new images as fire, smoke, or nonfire/ smoke. This involves providing the input image to the trained algorithm and allowing it to make a prediction based on the patterns and features it has learned from the dataset during training. To classify a new image using the Inception V3 model, we first need to upload the image into the algorithm. This can be done through a user interface that allows the user to select the image and upload it into the system. Once the image is uploaded, the algorithm processes it and makes a prediction based on the input.

The prediction process begins with the convolutional base of the Inception V3 architecture, which processes the image by extracting relevant features and patterns. The convolutional layers of the architecture are responsible for identifying the basic shapes and edges in the image, while the higher level layers are responsible for identifying more complex patterns such as textures and shapes. After the convolutional layers have processed the input image, the output is fed into the classification head of the model. The global average pooling layer calculates the average of the feature maps across the spatial dimensions, which reduces the dimensionality of the input and extracts the most important features. The output of the global average pooling layer is then fed into the dense layer, which performs the final classification of the image.

The dense layer consists of a fully connected layer that receives the input from the global average
pooling layer and performs a series of mathematical operations to calculate the probability of the image belonging to each class (fire, smoke, or non-fire/smoke). The ReLU activation function introduces non-linearity into the model, which allows it to model complex relationships between the input and output.
Finally, the softmax activation function is used to produce a probability distribution over the output classes, which gives the probability of the image belonging to each class.

Once the probability distribution is calculated, the algorithm makes a prediction by selecting the class with the highest probability. If the highest probability is associated with the "fire" class, the algorithm classifies the image as a fire. Similarly, if the highest probability is associated with the "smoke" class, the algorithm classifies the image as smoke. If the highest probability is associated with the "nonfire/ smoke" class, the algorithm classifies the image as non-fire/smoke.

# Testing And Evaluation

The testing and evaluation stage is a crucial component of the fire and smoke detection process, as it allows us to assess the performance of the Inception V3 model in detecting fires and smoke. In
this stage, we use a test dataset that was not used during the training phase to evaluate the accuracy of the model.

The test dataset consists of a set of images that were not used during the training phase. These images are labelled as fire, smoke, or non-fire/smoke, and are used to assess the accuracy of the model in classifying images. The performance of the model is evaluated by calculating the precision, recall, and F1-score.

If the accuracy of the model in detecting fires is not satisfactory, the test dataset may be updated to include more images of fires. This updated dataset is then used to retrain the model, which can improve its accuracy in detecting fires.

The process of updating the test dataset and retraining the model is known as iterative learning. In this process, the model is trained and tested multiple times using different datasets until the desired level of accuracy is achieved. This process helps to improve the performance of the model over time, and ensures that it can accurately detect fires and smoke in a range of different environments.

# Results and Description

The dataset we collected include images of fire and smoke, as well as non-fire and smoke images around 900 each for training the model and for testing and validation we have 100 images each. The collected dataset then scrubbed to remove any irrelevant or incorrect data, ensuring that the dataset is clean and suitable for training a machine learning model.

Once the dataset has been cleaned, we used it to train machine learning model to detect fire and smoke. Machine learning algorithm such as convolutional neural networks (CNNs) are commonly used for image recognition tasks, such as fire and smoke detection. It is trained using the cleaned dataset to identify patterns and features that distinguish fire and smoke from non-fire and smoke images. Validation and testing sets have been combined into one. Our dataset largely consists of three classes. Fire, Neutral, and Smoke are them. Each class has approximately 100 images for testing and over 900 images for training.
