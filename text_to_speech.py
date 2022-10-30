from gtts import gTTS
import os
from io import BytesIO
import datetime
def txt2speech(text):
    language = 'en'
    myobj = gTTS(text=text, lang=language, slow=False) 
    str1 = str(datetime.datetime.now())
    temp = ""
    for i in str1:
        if i not in [' ','-','.',':']:
            temp = temp + i
    audio_file_name = temp + "audio.mp3"
    myobj.save(audio_file_name) 
    return audio_file_name