import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np

from motion_autoencoder import MotionAutoencoder
from entropy_modeling import EntropyModel
from motion_estimation import MotionEstimation

class TrainAndVisualize:
    def __init__(self, autoencoder, flow, epochs=200, beta=0.01):
        """
        Initialize with your autoencoder model and flow data.
        
        Parameters:
        - autoencoder: Your trained autoencoder model.
        - flow: The flow data to be processed.
        - epochs: Number of training epochs.
        - beta: Weight for the entropy term in the loss.
        """
        self.autoencoder = autoencoder
        self.flow = tf.convert_to_tensor(flow, dtype=tf.float32)
        self.epochs = epochs
        self.beta = beta

        # Containers for training statistics
        self.losses = []
        self.rates = []
        self.dists = []
        self.q_steps = []

    def train(self):
        """
        Train the autoencoder using magnitude-based distortion.
        Prints progress every 10 epochs.
        Returns the final decoded output and entropy.
        """
        optimizer = tf.keras.optimizers.Adam(0.001)

        # Calculate flow magnitude for normalization (unused in loss but maybe for future improvements)
        flow_magnitude = tf.norm(self.flow, axis=-1)
        max_magnitude = tf.reduce_max(flow_magnitude)

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                decoded, entropy = self.autoencoder(self.flow, training=True)

                # Calculate magnitude-based distortion
                orig_magnitude = tf.norm(self.flow, axis=-1)
                recon_magnitude = tf.norm(decoded, axis=-1)
                distortion = tf.reduce_mean(tf.abs(orig_magnitude - recon_magnitude))

                rate = tf.reduce_mean(entropy)
                loss = distortion + self.beta * rate

            grads = tape.gradient(loss, self.autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))

            self.losses.append(loss.numpy())
            self.rates.append(rate.numpy())
            self.dists.append(distortion.numpy())
            self.q_steps.append(self.autoencoder.quantization_step.numpy())

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}:")
                print(f"  Loss: {loss:.4f}")
                print(f"  Distortion: {distortion:.4f}")
                print(f"  Rate: {rate:.4f}")
                print(f"  Quant Step: {self.autoencoder.quantization_step.numpy():.4f}")
                print(f"  Decoded Range: [{tf.reduce_min(decoded):.4f}, {tf.reduce_max(decoded):.4f}]")
                print(f"  Flow Range: [{tf.reduce_min(self.flow):.4f}, {tf.reduce_max(self.flow):.4f}]")
                
        return decoded, entropy

    def visualize(self, original, reconstructed, entropy):
        """
        Generate and save plots for the flow magnitudes, error map, entropy map,
        training curves, and quantization step evolution.
        
        Parameters:
        - original: Original flow data.
        - reconstructed: Decoded (reconstructed) flow data.
        - entropy: Entropy tensor from the model.
        """
        # Flow Comparison Plots
        plt.figure(figsize=(18, 6))
        
        # Original Flow Magnitude
        plt.subplot(1, 4, 1)
        orig_mag = np.sqrt(original[0,...,0]**2 + original[0,...,1]**2)
        vmax = np.percentile(orig_mag, 95)
        plt.imshow(orig_mag, cmap='jet', vmin=0, vmax=vmax)
        plt.title("Original Flow Magnitude")
        plt.colorbar()
        
        # Reconstructed Flow Magnitude
        plt.subplot(1, 4, 2)
        recon_mag = np.sqrt(reconstructed[0,...,0]**2 + reconstructed[0,...,1]**2)
        plt.imshow(recon_mag, cmap='jet', vmin=0, vmax=vmax)
        plt.title("Reconstructed Flow Magnitude")
        plt.colorbar()
        
        # Error Map
        plt.subplot(1, 4, 3)
        error = np.abs(orig_mag - recon_mag)
        plt.imshow(error, cmap='hot', vmin=0, vmax=np.percentile(error, 95))
        plt.title("Absolute Error")
        plt.colorbar()
        
        # Entropy Map
        plt.subplot(1, 4, 4)
        entropy_map = entropy.numpy().squeeze().mean(-1)
        plt.imshow(entropy_map, cmap='viridis')
        plt.title("Entropy Map")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('flow_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Training Curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.losses)
        plt.title("Total Loss")
        
        plt.subplot(1, 3, 2)
        plt.plot(self.dists)
        plt.title("Distortion")
        
        plt.subplot(1, 3, 3)
        plt.plot(self.rates)
        plt.title("Rate")
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Quantization Step Evolution
        plt.figure()
        plt.plot(self.q_steps)
        plt.title("Quantization Step Evolution")
        plt.savefig('quantization_step.png', dpi=300, bbox_inches='tight')
        plt.close()


#trainer = TrainAndVisualize(autoencoder, flow, epochs=200, beta=0.01)
#decoded, entropy = trainer.train()
#trainer.visualize(flow, decoded, entropy)
