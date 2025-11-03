import sensor
import time
import ml

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))  # Set 240x240 window.
sensor.skip_frames(time=2000)  # Let the camera adjust.

with open('labels.txt', 'r') as file:
    labels = [line.strip() for line in file if line.strip()]

model = ml.Model("trained.tflite", load_to_fb=True)
norm = ml.Normalization(scale=(-1.0, 1.0))

clock = time.clock()
while True:
    clock.tick()
    img = sensor.snapshot()
    input = [norm(img)] # scale 0~255 to -1.0~1.0
    scores = model.predict(input)[0].flatten().tolist()
    for label, score in zip(labels, scores):
        print(label, score)
    print(clock.fps(), "fps")