import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from googletrans import LANGUAGES, Translator
import speech_recognition as sr


# Function to translate text
def translate_text(text, source_lang, target_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=target_lang)
    return translated_text.text


# Function to generate image from text
def generate_images_from_text(text, num_images=1, base_iteration=0.1):
    # Use the text-to-image model to generate the image
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
            return None

    return images


# Function to transcribe speech input
def transcribe_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=10)  # Adjust timeout value as needed
        try:
            text = r.recognize_google(audio)
            st.write("You said:", text)
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I didn't catch that. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None
    
    except OSError as e:
        st.error(f"Microphone not accessible: {e}")
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
    if st.button("Generate Image from Speech"):
        speech_text = transcribe_speech()
        if speech_text:
            translated_text = translate_text(speech_text, "en", target_lang)
            images = generate_images_from_text(translated_text, num_images=num_images)
            if images:
                st.write(f"English Translated Prompt: ", translated_text)
                for i, image in enumerate(images):
                    st.image(image, caption=f"Generated Image {i + 1}", use_column_width=True)
            else:
                st.error("Failed to generate images. Please try again.")


if __name__ == "__main__":
    main()
