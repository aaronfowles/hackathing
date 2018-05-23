from io import BytesIO, open

import numpy as np
from scipy.io.wavfile import write
import pyaudio
from google.cloud import speech, language
from google.cloud.speech import enums
from google.cloud.speech import types
from sense_hat import SenseHat

pyaud = pyaudio.PyAudio()

stream = pyaud.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=48000,
    input_device_index=2,
    input=True
)

list_amplitude_array = list()

for i in range(0, (48000 / 1024) * 5):
    rawsamps = stream.read(1024, exception_on_overflow=False)
    samps = np.fromstring(rawsamps, dtype=np.int16)
    list_amplitude_array.append(samps)

stream.close()

audio_signal = np.concatenate(list_amplitude_array)
amplified_audio_signal = audio_signal * 10
audio_buffer = BytesIO()
write(audio_buffer, 48000, amplified_audio_signal)
audio_buffer.seek(0)
client = speech.SpeechClient()
content = audio_buffer.read()
audio = types.RecognitionAudio(content=content)
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=48000,
    language_code='en-US')

response = client.recognize(config, audio)

total_results = list()
for result in response.results:
        total_results.append(result.alternatives[0].transcript)

text = ' '.join(total_results)

language_client = language.LanguageServiceClient()

document = language.types.Document(
    content=text,
    type=language.enums.Document.Type.PLAIN_TEXT)
response = language_client.analyze_sentiment(document=document)

score = response.document_sentiment.score

sense = SenseHat()

if score > 0.5:
    sense.show_letter(")", text_colour=[0,255,0])
elif score < 0.5:
    sense.show_letter("(", text_colour=[0,0,255])
else:
    sense.show_letter("|", text_colour=[100,100,100])