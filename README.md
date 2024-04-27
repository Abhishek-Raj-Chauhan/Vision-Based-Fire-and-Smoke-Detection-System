# Vision-Based Fire and Smoke Detection System
 The aim of this project is real-time and accurate identification of fire and smoke occurrences to enhance disaster management and emergency response strategies.Utilized Convolutional Neural Networks (CNNs) with InceptionV3 and AlexNet architectures, along with data preprocessing, data augmentation, multiclass classification with softmax, ReLU activation, evaluation metrics such as accuracy, and frameworks like Keras and TensorFlow for efficient model development and deployment.

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
