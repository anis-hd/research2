import tensorflow as tf
import tensorflow_addons as tfa
from .entropy_model import EntropyModel

class MotionAutoencoder(tf.keras.Model):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

        # Enhanced Encoder with Instance Normalization
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
        ])

        # Instantiate the EntropyModel
        self.entropy_model = EntropyModel()

        # Enhanced Decoder with Instance Normalization
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(2, 3, 1, 'same')
        ])

        # Smaller quantization step to start
        self.quantization_step = tf.Variable(0.01, trainable=True)

    def quantize(self, x):
        # Adds some rounding noise... because perfection is overrated.
        return x + tf.stop_gradient(tf.round(x / self.quantization_step) * self.quantization_step - x)

    def call(self, mv, training=False):
        # Normalize input motion vectors to [-1, 1]
        mv_magnitude = tf.norm(mv, axis=-1, keepdims=True)
        mv_normalized = mv / (tf.reduce_max(mv_magnitude) + 1e-8)

        # Encode and quantize
        compressed = self.encoder(mv_normalized)
        quantized = self.quantize(compressed)

        # Compute entropy using our separate entropy model
        entropy = self.entropy_model(quantized, self.quantization_step)

        # Decode and denormalize
        decoded = self.decoder(quantized)
        decoded = decoded * tf.reduce_max(mv_magnitude)  # scale back to original range

        return decoded, entropy