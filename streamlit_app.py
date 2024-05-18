import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from googletrans import LANGUAGES, Translator
from audio_recorder_streamlit import audio_recorder
import os

# Function to translate text
def translate_text(text, source_lang, target_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=target_lang)
    return translated_text.text

# Function to generate image from text
def generate_images_from_text(text, num_images=1, base_iteration=0.1):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    images = []
    for i in range(num_images):
        iteration = base_iteration * (i + 1)  # Adjust iteration for each image
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

# Function to transcribe speech input from audio file using Google Speech Recognition API
def transcribe_speech_from_audio_file(audio_file_path):
    import speech_recognition as sr

    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            st.write("Transcribed audio:", text)
            return text
    except Exception as e:
        st.error(f"Error transcribing audio file: {e}")
        return None

def main():
    st.title("Multilingual Image Generation Using Text & Speech")

    # Get all available languages supported by Google Translate
    available_languages = list(LANGUAGES.values())

    # Sidebar for language selection
    source_lang = st.sidebar.selectbox("Select Source Language", available_languages)
    target_lang = "en"  # Fixed target language as English for Stable Diffusion

    # Text input
    text_input = st.text_area("Enter text in the source language")

    # Number of images to generate
    num_images = st.number_input("Number of Images to Generate", min_value=1, max_value=10, value=1, step=1)

    # Generate image from text
    if st.button("Generate Images from Text"):
        if text_input:
            translated_text = translate_text(text_input, source_lang, target_lang)
            images = generate_images_from_text(translated_text, num_images=num_images)
            if images:
                st.write(f"English Translated Prompt: ", translated_text)
                for i, image in enumerate(images):
                    st.image(image, caption=f"Generated Image {i + 1}", use_column_width=True)
            else:
                st.error("Failed to generate images. Please try again.")

    # Speech input
    st.button("Generate Images from Speech")
    audio_bytes = audio_recorder(text="Tap on the mic to start recording:", icon_size="2x")

    if audio_bytes:
        file_name = "speech_recorded.wav"
        with open(file_name, "wb") as audio_file:
            audio_file.write(audio_bytes)

        speech_text = transcribe_speech_from_audio_file(file_name)
        os.remove(file_name)

        if speech_text:
            translated_text = translate_text(speech_text, source_lang, target_lang)
            images = generate_images_from_text(translated_text, num_images=num_images)
            if images:
                st.write(f"English Translated Prompt: ", translated_text)
                for i, image in enumerate(images):
                    st.image(image, caption=f"Generated Image {i + 1}", use_column_width=True)
            else:
                st.error("Failed to generate images. Please try again.")
        else:
            st.error("Speech transcription failed. Please try again.")

if __name__ == "__main__":
    main()
