import speech_recognition as sr
from gtts import gTTS
from pygame import mixer

# for speech recognition from microphone
recognizer = sr.Recognizer()

# to convert text back into robotic speech
pygame.init()
screen = pygame.display
mixer.init()

while (True): 

    with sr.Microphone() as source: 
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

        speech = gTTS(text)
        speech.save("response.mp3")
        mixer.music.load("response.mp3")
        mixer.music.play()









