
import tensorflow as tf
from models import build_generator, build_discriminator

IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 16
EPOCHS = 5

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    labels=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

dataset = dataset.map(
    lambda x: (tf.cast(x, tf.float32) / 127.5) - 1
).prefetch(tf.data.AUTOTUNE)

generator = build_generator(LATENT_DIM)
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)


@tf.function
def train_step(real_images):
    noise = tf.random.normal((tf.shape(real_images)[0], LATENT_DIM))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        gen_loss = cross_entropy(
            tf.ones_like(fake_output), fake_output
        )

        real_loss = cross_entropy(
            tf.ones_like(real_output), real_output
        )

        fake_loss = cross_entropy(
            tf.zeros_like(fake_output), fake_output
        )

        disc_loss = real_loss + fake_loss

    gen_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )

    disc_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


for epoch in range(EPOCHS):
    for images in dataset:
        gen_loss, disc_loss = train_step(images)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Generator loss: {gen_loss:.4f} "
        f"Discriminator loss: {disc_loss:.4f}"
    )

    if (epoch + 1) % 10 == 0:
        generator.save_weights("weights/generator.weights.h5")

generator.save_weights("weights/generator.weights.h5")
print("GAN training completed!")
