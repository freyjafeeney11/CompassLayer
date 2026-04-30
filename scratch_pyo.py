import pyo
import time
import os

print("Booting pyo...")
s = pyo.Server(audio="portaudio", duplex=0).boot()
s.start()

print("Loading sound...")
path = "c:/Users/Freyja/Desktop/shadows/CompassLayer/assets/audio/koto_note.wav"
if not os.path.exists(path):
    print("WAV not found")
else:
    sf = pyo.SfPlayer(path, loop=False)
    sf_mono = sf.mix(1) # Mix to mono for binaural
    pitch = pyo.Harmonizer(sf_mono, transpo=0, winsize=0.05)
    bin = pyo.Binaural(pitch, azimuth=0, elevation=0).out()

    print("Playing center...")
    sf.play()
    time.sleep(1)
    
    print("Playing center pitched up...")
    pitch.setTranspo(1)
    sf.play()
    time.sleep(1)

    print("Playing right...")
    pitch.setTranspo(0)
    bin.setAzimuth(90)
    sf.play()
    time.sleep(1)

s.stop()
print("Done")
