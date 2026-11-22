
import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder():
    inputs = layers.Input(shape=(128, 128, 3))

    # Encoder
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    # Decoder
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    outputs = layers.Conv2D(
        3, 3, activation="sigmoid", padding="same"
    )(x)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return autoencoder


def build_generator(latent_dim=100):
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(
            128, 5, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Conv2DTranspose(
            64, 5, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Conv2DTranspose(
            32, 5, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Conv2DTranspose(
            3, 5, strides=2, padding="same",
            activation="tanh"
        )
    ])

    return model


def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Conv2D(64, 5, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 5, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 5, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])

    return model
