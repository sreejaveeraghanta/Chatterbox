import speech_recognition as sr
from gtts import gTTS
import pygame
from openai import OpenAI
from pygame import mixer
from model import predict
import numpy as np
import threading

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

client = OpenAI()

def get_response(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=[{'role': "user", "content": prompt}], 
        n = 1,
        max_tokens=100, 
        stop=None,
        temperature=0.7)
    return response.choices[0].message.content.strip()

def listen_and_record():
    while True:
        with sr.Microphone(device_index=mic_ind) as source: 
            recognizer.adjust_for_ambient_noise(source)
            recognizer.energy_threshold = 400
            try: 
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=30) 
                with open('input.wav', "wb") as input_audio:
                    input_audio.write(audio.get_wav_data()) 

                text = recognizer.recognize_google(audio)
                print("The text from you", text)
            except: 
                print("wating for you to say something")
                continue

            response = get_response(text)

            speech = gTTS(response, slow=False)
            speech.save("response.mp3")
            mixer.music.load("response.mp3")
            mixer.music.play()


thread_listen_record = threading.Thread(target=listen_and_record, daemon=True)
thread_listen_record.start()

running = True
while (running): 
    ## pick a different model for prediction by changing the second argument
    ## possible model_types "ravdess", "crema", "both"
    emotion = predict('input.wav', model_type="ravdess")
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            running = False
    screen.fill((0,0,0))
    if (emotion == "anger"):
        image = pygame.image.load('./emojis/anger.png')
        image = pygame.transform.scale(image, (400, 400))
    elif (emotion == "happy"):
        image = pygame.image.load('./emojis/happy.png')
        image = pygame.transform.scale(image, (400, 400))
    elif (emotion == "sad"):
        image = pygame.image.load('./emojis/sad.png')
        image = pygame.transform.scale(image, (400, 400))
    elif (emotion == "surprised"):
        image = pygame.image.load('./emojis/surprised.png')
        image = pygame.transform.scale(image, (400, 400))
    elif (emotion == "disgust"):
        image = pygame.image.load('./emojis/disgust.png')
        image = pygame.transform.scale(image, (400, 400))
    elif (emotion == "fear"):
        image = pygame.image.load('./emojis/fear.png')
        image = pygame.transform.scale(image, (400, 400))
    else: 
        image = pygame.image.load('./emojis/neutral.png')
        image = pygame.transform.scale(image, (400, 400))

    screen.blit(image, (100, 100))
    pygame.display.flip()
pygame.quit()





