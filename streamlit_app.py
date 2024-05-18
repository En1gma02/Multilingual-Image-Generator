import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from googletrans import LANGUAGES, Translator
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder

# Function to translate text
def translate_text(text, source_lang, target_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=target_lang)
    return translated_text.text

# Function to generate images from text
def generate_images_from_text(text, num_images=1, base_iteration=0.1):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    images = []
    for i in range(num_images):
        iteration = base_iteration * (i + 1)
        payload = {"inputs": text, "iteration": iteration}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
        else:
            st.error(f"Failed to generate image {i + 1}. HTTP status code: {response.status_code}")
            return None

    return images

# Function to transcribe speech input from microphone
def transcribe_speech_from_microphone(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        st.write("Sorry, I didn't catch that. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None

def main():
    st.title("Multilingual Image Generation Using Text & Speech")

    available_languages = list(LANGUAGES.values())

    source_lang = st.sidebar.selectbox("Select Source Language", available_languages)
    target_lang = "en"

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        text_input = st.text_area("Enter text in the source language")
    with col2:
        audio_bytes = audio_recorder(text="Or Record Your Speech", icon_size="2x")

        if audio_bytes:
            transcribed_text = transcribe_speech_from_microphone(audio_bytes)
            if transcribed_text:
                text_input = transcribed_text
                st.experimental_rerun()

    num_images = st.number_input("Number of Images to Generate", min_value=1, max_value=10, value=1, step=1)

    if st.button("Generate Images"):
        if text_input:
            translated_text = translate_text(text_input, source_lang, target_lang)
            images = generate_images_from_text(translated_text, num_images=num_images)
            if images:
                st.write(f"English Translated Prompt: ", translated_text)
                for i, image in enumerate(images):
                    st.image(image, caption=f"Generated Image {i + 1}", use_column_width=True)
            else:
                st.error("Failed to generate images. Please try again.")

if __name__ == "__main__":
    main()
