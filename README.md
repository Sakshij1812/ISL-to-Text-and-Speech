# ISL-to-Text-and-Speech
This is a python project that uses CNN to extract Indian Sign Language from real-time video-feed and convert it to text and speech.

CreateDataset.py -->
I used a dataset downloaded from Kaggle : https://www.kaggle.com/vaishnaviasonawane/indian-sign-language-dataset
But decided to add more to the dataset by creating my own images and saving it to the train/test/val files. 
Different files need to be creeated for different signs. So the filename needs to be updated in the code for each Sign (Class)

TrainCNN_Model.py -->
Model is trained using CNN and saved in an H5 file called best_model.h5 and class indices is stored in classes.npy

VideoCapture.py -->
This program uses the computer camera to capture the image from ROI (Region of Interest - depicted by a green square on the video-feed) and predict the sign. 
It is then converted to speech using pyttsx3 module in python.
The prediction is made with the help of the model and the class indices saved while training model in TrainCNN_Model.py
