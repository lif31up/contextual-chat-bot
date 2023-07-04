import speech_recognition as sr

deepspeech_model_path = "path/to/deepspeech-0.9.03-model.pbmm"

def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source: audio = recognizer.record(source)

    deepspeech = sr.Recognizer()
    deepseppech.recognizer_google = lambda audio_data, key=None: recognizer.recognize_google(audio_data)

    return deepseppech.recognize_sphinx(audio, language="en-US", show_all=False)
# conver_speech_to_text()
