import speech_recognition as sr
from gtts import gTTS
import pygame
from openai import OpenAI
from pygame import mixer
from model import *
import numpy as np
import librosa
def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, hop_length=512):
    """Extracts a Mel spectrogram from an audio file using librosa."""
    #loading the audio files 
    y, sr = librosa.load(file_path, sr=sr)
    #compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    #power to db
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) 
    if mel_spec_db.shape[1] > 128: 
        mel_spec_db = mel_spec_db[:, :128]
    if mel_spec_db.shape[1] < 128: 
        padding = 128 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, padding)), mode='constant')
    return mel_spec_db
# for speech recognition from microphone
recognizer = sr.Recognizer()

# to convert text back into robotic speech
pygame.init()
screen = pygame.display
mixer.init()

for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(i,":", name)

mic_ind = int(input("select microphone index from list: "))
file = './1001_IOM_SAD_XX.wav'
features = torch.tensor(extract_mel_spectrogram(file), dtype=torch.float32)

emotion = predict(features)
print(emotion)

client = OpenAI()

def get_response(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=[{'role': "user", "content": prompt}])
    return response.choices[0].message.content.strip()

while (True): 
    with sr.Microphone(device_index=mic_ind) as source: 
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source) 
        try: 
            text = recognizer.recognize_google(audio)
            print("The text from you", text)
        except: 
            print("wating for you to say something")
            continue
        if  text == "stop": 
            break

        response = get_response(text)

        speech = gTTS(response)
        speech.save("response.mp3")
        mixer.music.load("response.mp3")
        mixer.music.play()




