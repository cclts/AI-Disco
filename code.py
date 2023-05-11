import os, vlc, time, random, librosa
import numpy as np
from gpiozero import LED
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite

GENRES = ['metal', 'disco', 'classical', 'hippop', 'jazz',
          'country', 'pop', 'blues', 'reggae', 'rock']
COLORS = [(35,30,35), (61,39,0), (53,44,3), (0,0,100), (39,30,31),
          (66,17,17), (23,36,41), (0,0,100), (0,100,0), (73,7,20)]

SAMPLE_SIZE = 660000
path = "/home/pi/Music/"
playlist = os.listdir(path)

led = LED(18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

pwm0 = GPIO.PWM(12, 200)
pwm0.start(0)
pwm1 = GPIO.PWM(13, 200)
pwm1.start(0)

def beat_track(signal):
    _, beats = librosa.beat.beat_track(y=signal, tightness=100)
    beat_times = librosa.frames_to_time(beats)
    beat_times = beat_times.astype(np.float16)
    duration = []
    for i in range(beat_times.shape[0]):
        if i == 0:
            duration.append(beat_times[0])
        else:
            d = beat_times[i] - beat_times[i-1]
            duration.append(d)
    return duration

def split_song(signal):
    y = []
    window_size = int(signal.shape[0] * 0.1)
    y.append(signal[1:window_size])
    return np.array(y)

def generate_spectrograms(signals):
    rep = []
    for instance in signals:
            rep.append(librosa.feature.melspectrogram(instance))
    return np.array(rep)

try:
    print('press Ctrl-C to stop')
    while True:
        for song in playlist:
            player = vlc.MediaPlayer(path + song)
            signal, _ = librosa.load(path + song)
            
            #generate input data
            middle_sample = int(len(signal)/SAMPLE_SIZE/2)
            middle_part = signal[middle_sample * SAMPLE_SIZE: (middle_sample+1) * SAMPLE_SIZE]
            window = split_song(middle_part)
            middle_spectr = generate_spectrograms(window)
            
            interpreter = tflite.Interpreter(model_path="model.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test model on random input data.
            input_shape = input_details[0]['shape']
            input_data = np.array(middle_spectr, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = output_data[0].tolist()
            
            flag = True
            duration = beat_track(signal)
            player.play()
            print("playing " + song)
            
            for d in duration:
                effect = random.choices(range(10), output_data)[0]
                print('special effect is ' + GENRES[effect])
                print('last for ' + str(d.item()) + ' sec\n')
                pwm0.ChangeDutyCycle(COLORS[effect][1])
                pwm1.ChangeDutyCycle(COLORS[effect][2])
                if COLORS[effect][0] > 50:
                    led.on()
                else:
                    led.off()
                time.sleep(d.item())
except KeyboardInterrupt:
    print('stop')
finally:
    pwm0.stop()
    pwm1.stop()
    GPIO.cleanup()