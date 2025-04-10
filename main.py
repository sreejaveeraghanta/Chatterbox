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
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("CHATTERBOX")
mixer.init()

for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(i,":", name)

mic_ind = int(input("select microphone index from list: "))
file = './1002_DFA_ANG_XX.wav'
features = torch.tensor(extract_mel_spectrogram(file), dtype=torch.float32)
print(features)

emotion = predict(features)
print(emotion)

client = OpenAI()

def get_response(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=[{'role': "user", "content": prompt}])
    return response.choices[0].message.content.strip()

running = True
while (running): 
    # for event in pygame.event.get(): 
    #     if event.type == pygame.QUIT: 
    #         running = False
    # screen.fill((0,0,0))
    if (emotion == "angry"):
        image = pygame.image.load('./emojis/anger.png')
        image = pygame.transform.scale(image, (400, 400))

    else: 
        image = pygame.image.load('./emojis/neutral.png')
        image = pygame.transform.scale(image, (400, 400))

    # screen.blit(image, (100, 100))
    # pygame.display.flip()
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
            running = False
            break

        response = get_response(text)

        speech = gTTS(response, slow=False)
        speech.save("response.mp3")
        mixer.music.load("response.mp3")
        mixer.music.play()
pygame.quit()





