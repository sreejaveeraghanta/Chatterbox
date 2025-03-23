import speech_recognition as sr
import pyttsx3
import pyaudio

# for speech recognition from microphone
recognizer = sr.Recognizer()

# to convert text back into robotic speech
# TODO turn this into the expressive voice later
 speech_engine = pyttsx3.init()

while (True): 

    with sr.Microphone() as source1: 
        audio = recognizer.listen(source1) 
        try: 
            text = recognizer.recognize_google(audio)
            print("The text from you", text)

        except: 
            print("wating for you to say something")
            continue

        if  text == "stop": 
            break

        speech_engine.say(text) 
        speech_engine.runAndWait()








