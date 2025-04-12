# Install Dependencies and Packages 
```bash
$ python3 -m venv <env_name> 
$ source <env_name>/bin/activate 
$ pip3 install -r requirements.txt 
```
if on a MacOS: <br> 
```bash
$ brew install portaudio 
```
before installing requirements


# Run the program
our project uses the OpenAI API to generate responses from a GPT model, so an API key is needed to run the main program (main.py)<br>
```bash
$ python3 main.py
```
The model might need to be trained again to run the program due to large file size, we could not include them in the submissions, can be trained with the following: 
```bash
$ python3 traing.py
```


# Structure
## Data 
We got our datasets from Kaggle: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en <br>
From these datasets we used: RAVDESS and CREMA-D to train our model. These datasets are stored in the "data" folder under their corresponding folder names: 
- Ravdess:
  - audio_speech_actors_01-24
    - Actor_01/
      - 03-01....wav
    - ...
- Crema:
  - 1001_....wav
  - ...
 
The data was stored in a similar manner to how we got it from kaggle, but we did preprocessing on the data to get a dataframe with files, labels and emotions from these file identifiers. <br>

Our code is split into corresponding files: <br>
- main.py: this is the main program that can be run to get the application
  - This main application uses one of the saved models to predict emotions based on real time data being recorded by a microphone
- training.py: this file contains the training and testing done on each of the datasets that we used
- preprocessing.py: this file contains functions used to process and load the data so that it can be used by the model
- data_loading.py: this file contains functions to extract features and load datsets into dataloaders to be used by the model
- model.py: this file contains the model architecture and the corresponding functions associated with the model
- models: folder that stores all the saved models from tarining
- emojis: folder containing all the emojis that can be displayed by the application

# Evaluation
Our model did not perform as well as we hoped, as we happen to have a bit of overfittinh. Therefore, the application is unable to accurately detect a few of the emotions. The application also picks up on noise, so it is randomly displaying emojis even when nothing is being said. This is something that we will need to look into fixing in the future. <br> 
Our project initially was going to display an animated character instead of an emoji, but we decided to switch to displaying an emoji because training the model to receive good results took a very long time. We were able to bring our model's accuracy up quite a bit from our initial stages of training, however there is still a bit of overfitting we will still need to mitigate<br>

