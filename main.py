import speech_recognition as sr
from gtts import gTTS
import pygame
from openai import OpenAI
from pygame import mixer
from model import *

# for speech recognition from microphone
recognizer = sr.Recognizer()

# to convert text back into robotic speech
pygame.init()
screen = pygame.display
mixer.init()

for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(i,":", name)

mic_ind = int(input("select microphone index from list: "))

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

        emotion = predict(text)
        print(emotion)

        response = get_response(text)

        speech = gTTS(response)
        speech.save("response.mp3")
        mixer.music.load("response.mp3")
        mixer.music.play()




