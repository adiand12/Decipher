from gtts import gTTS
import os

def txt2speech(text):
    # The text that you want to convert to audio
    # Language in which you want to convert
    language = 'en'
    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    myobj = gTTS(text=text, lang=language, slow=False)
    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save("DecipheredAudio.mp3")
    # Playing the converted file
    os.system("mpg321 DecipheredAudio.mp3")