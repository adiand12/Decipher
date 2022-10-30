import streamlit as st
from streamlit_option_menu import option_menu
from gtts import gTTS
from text_to_speech import txt2speech
from translation import translator
from io import BytesIO
from imgtotxt import img2txt
st.set_page_config(layout="wide")
col1,col2 = st. columns(2)
#Hindi_text=""
text=""
Eng_text=""
zx=BytesIO()
x=False
with col1:
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
    Decipher is an image-text extraction and translation system. So why wait? upload an image right away!  ''')
    st.header('Upload Below')
    image = st.file_uploader("Upload an image containing Hindi text, happy deciphering :)")
    if image is not None:
        # To read file as bytes:
        data = image.read()
    if st.button('Submit'):
        text = img2txt(data)
        Eng_text = translator(text, 'Hindi')
        name = txt2speech(Eng_text)
        x=True
        
#st.text_area('image name', str(image.name))
with col2:    
    st.header('Results:')
    if x:
        Hindi_text = st.text_area('Hindi text:',text) 
        st.text_area('English text',Eng_text)
        #st.download_button("Download Audio",zx)

        audio_file = open(name, 'rb')
        audio_bytes = audio_file.read()
        #st.audio(audio_bytes, format='audio/,p3')
        st.write("Listen Here:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)


