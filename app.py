import streamlit as st
from streamlit_option_menu import option_menu
from gtts import gTTS
from text_to_speech import txt2speech
from translation import translator
from imgtotxt import img2txt

selected = option_menu(
            menu_title="dEchipHer", 
            options=[" "],  # required
            menu_icon="blockquote-left",  # optional
            default_index=0,  # optional
            orientation="horizontal",

            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "18px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "black"},
            },
        )

st.title("Decipher: A Hindi OCR English Translation and Speech Conversion System")
st.write(''' 
Decipher is an image-text extraction and translation system. So why wait? upload an image iPhone right away!  ''')
st.header('Upload Below')
image = st.file_uploader("Upload an image containing Hindi text, happy deciphering :)")
if image is not None:
     # To read file as bytes:
     data = image.read()

#st.text_area('image name', str(image.name))

if st.button('Submit'):
    text = img2txt(data)
    Hindi_text = st.text_area('Hindi text:',text) 
    Eng_text = translator(Hindi_text, 'Hindi')
    st.text_area('English text',Eng_text)
    txt2speech(Eng_text)
    st.download_button("Download Audio",'DecipheredAudio.mp3')
 
    
