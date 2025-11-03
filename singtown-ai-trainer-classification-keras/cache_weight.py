# cache: ~/.keras/models

import keras

for alpha in [0.25, 0.50, 0.75, 1.0]:
    for imgsize in [128, 160, 192, 224]:
        try:
            keras.applications.MobileNet(
                alpha=alpha,
                include_top=False,
                input_shape=(imgsize, imgsize, 3),
                weights="imagenet",
            )
        except Exception:
            print("download error MobileNet", alpha, imgsize)


for alpha in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
    for imgsize in [96, 128, 160, 192, 224]:
        try:
            keras.applications.MobileNetV2(
                alpha=alpha,
                include_top=False,
                input_shape=(imgsize, imgsize, 3),
                weights="imagenet",
            )
        except Exception:
            print("download error MobileNetV2", alpha, imgsize)
