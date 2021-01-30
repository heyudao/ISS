# Emotion detection using deep learning

## Introduction
This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. 
The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). 
This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies
To set it up : 
1.) Anaconda command promote (create environment)
conda create -n prs python=3.7
2.) Activate environment 
conda activate prs
3.) Navigate to ur folder 
cd C:\Users\He Yudao\Desktop\NUS Masters\PatternRecognitionSystems\PM\Emotion-detection-master\Emotion-detection-master
4.) Install the required packages
pip install -r requirements.txt 
conda install spyder matplotlib SciPy (for editing in spyder)

## Training the model
cd C:\Users\He Yudao\Desktop\NUS Masters\PatternRecognitionSystems\PM\Emotion-detection-master\Emotion-detection-master\src
python emotions.py --mode train
this will output the model to model.h5

-----------------------------------------------
The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)
-----------------------------------------------

##Use Case 1 (save image)
python emotions.py --mode display_image (this mode detect the emotion of a single saved image in the folder)

##Use Case 2 (web cam live emotion detection)
python emotions.py --mode display_webcam (this mode detect the emotion of person in webcam live)
This implementation by default detects emotions on all faces in the webcam feed. 
With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.
* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.
* The network outputs a list of **softmax scores** for the seven classes of emotions.
* The emotion with maximum score is displayed on the screen.

-----------------------------------------------------------------------------------------------------------------------------------------------

## Data Preparation (optional)
The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. 
I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.
In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. 
I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## References
* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.
