import tensorflow as tf
import tensorflow_addons as tfa

class EntropyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Hyperprior networks for entropy estimation
        self.hyper_mean = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
        ])

        self.hyper_scale = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, 3, 1, 'same'),
        ])

    def call(self, quantized, quantization_step):
        mean = self.hyper_mean(quantized)
        scale = tf.nn.softplus(self.hyper_scale(quantized)) + 1e-6

        # Rate estimation using probability computed from the hyperprior
        upper = quantized + quantization_step / 2
        lower = quantized - quantization_step / 2
        cdf_upper = 0.5 * (1 + tf.math.erf((upper - mean) / (scale * tf.sqrt(2.0))))
        cdf_lower = 0.5 * (1 + tf.math.erf((lower - mean) / (scale * tf.sqrt(2.0))))
        prob = cdf_upper - cdf_lower
        entropy = -tf.math.log(prob + 1e-10) / tf.math.log(2.0)
        return entropy