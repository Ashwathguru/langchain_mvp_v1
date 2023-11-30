import os
import sys
import datetime
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

from bokeh.models.widgets import Button
from bokeh.models import CustomJS

# import API key from .env file
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_answer_csv(query: str) -> str:
    file = "raw.csv"
    agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False)
    answer = agent.run(query)
    return answer

def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    return transcript


def save_audio_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(file_path):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)

    return transcript["text"]



def main():
    """
    Main function to run the Whisper Transcription app.
    """
    st.title("TicketGPT")

    tab1, tab2 = st.tabs(["Speak", "Chat"])

    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, "mp3")
        # Transcribe button action
        if st.button("Submit Voice command ?"):
            # Find the newest audio file
            audio_file_path = max(
                [f for f in os.listdir(".") if f.startswith("audio")],
                key=os.path.getctime,
            )

            # Transcribe the audio file
            transcript_text = transcribe_audio(audio_file_path)

            # Display the transcript
            st.header("Transcript")
            st.write(transcript_text)
            query=transcript_text
            response=get_answer_csv(query)
            st.write(response)

            tts_button = Button(label="Talk to me", width=100)

            tts_button.js_on_event("button_click", CustomJS(code=f"""
                                    var u = new SpeechSynthesisUtterance();
                                    u.text = "{response}";
                                    u.lang = 'en-US';

                                    speechSynthesis.speak(u);
                                    """))

            st.bokeh_chart(tts_button)
            # Save the transcript to a text file
            with open("response.txt", "w") as f:
                f.write(response)

            # Provide a download button for the transcript
            st.download_button("Download Response", response)

    # Upload Audio tab
    with tab2:
        query = st.text_area("Ask any question related to the tickets")
        button = st.button("Submit")
        if button:
            response=get_answer_csv(query)
            st.write(response)
            # Save the transcript to a text file
            with open("response.txt", "w") as f:
                f.write(response)

            # Provide a download button for the transcript
            st.download_button("Download Response", response)

if __name__ == "__main__":
    # Set up the working directory
    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    # Run the main function
    main()