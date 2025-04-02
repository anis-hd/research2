import tensorflow as tf
import tensorflow_addons as tfa


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

        # Enhanced Decoder
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

        # Entropy modeling is now modular!
        self.entropy_model = EntropyModel()

        # Initialize with a smaller quantization step
        self.quantization_step = tf.Variable(0.01, trainable=True)

    def quantize(self, x):
        return x + tf.stop_gradient(tf.round(x / self.quantization_step) * self.quantization_step - x)

    def call(self, mv, training=False):
        # Normalize input to [-1, 1] range
        mv_magnitude = tf.norm(mv, axis=-1, keepdims=True)
        mv_normalized = mv / (tf.reduce_max(mv_magnitude) + 1e-8)

        # Encoding and quantization
        compressed = self.encoder(mv_normalized)
        quantized = self.quantize(compressed)

        # Entropy modeling via the modular class
        entropy = self.entropy_model(quantized, self.quantization_step)

        # Decoding and denormalization
        decoded = self.decoder(quantized)
        decoded = decoded * tf.reduce_max(mv_magnitude)  # Scale back to original range

        return decoded, entropy
