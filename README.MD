# TFJS-Image-Classifier-Simple-HTML
Implementation of Tensorflow.js using a simple static webpage and a custom ML model. An overview of the project can be found on [YouTube](https://youtu.be/n-e-zYiMcTo).

## To run
Download the files, then from the command line, create a local server:
```cmd
python -m http.server
```
Navigate to http://localhost:8000/index.html and http://localhost:8000/staticimage_index.html to see the resulting demos. For the webcam demo, you will be asked to share your webcam.

## Overview
### Data Source
For the creation of the model, I used the [Microsoft Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). This dataset contains 12,500 cat images and 12,500 dog images.

### Model Creation
Only a subset of the data was used to create the model (1,500 of each class). [Google Cloud's AutoML service](https://cloud.google.com/automl) was used to train the model then it was exported to files to be used in Tensorflow.js.

### First Demo: Static Images
The first demo in index.html uses the exported model to make predictions about five images. This is a pretty standard demo that shows the power of Tensorflow.js to make predictions at the time of image load.

### Demo #2: Webcam
In the second demo, located in webcam_index.js, uses the webcam to make predictions. This demo is only useful if you can point the camera to a nearby cat or dog. However, this shows how you could make continuous predictions using Tensorflow.js using the webcam as an edge device. 