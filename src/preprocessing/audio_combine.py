from pydub import AudioSegment

audio1 = AudioSegment.from_file("data/raw/musan/noise/sound-bible/noise-sound-bible-0004.wav")
audio2 = AudioSegment.from_file("data/raw/musan/speech/us-gov/speech-us-gov-0000.wav")

audio1 = audio1 - 3
# audio2 = audio2 - 6

mixed = audio1.overlay(audio2)

mixed.export("output_mix.wav", format="wav")

print("Done!")
