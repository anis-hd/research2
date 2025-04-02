import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

def warp_frame_with_motion_vectors(prev_frame_path, next_frame_path, autoencoder, motion_estimator, show_results=True):
    """
    Warps the previous frame using motion vectors estimated and compressed by the autoencoder.
    
    Args:
        prev_frame_path: Path to the previous frame (f001.png)
        next_frame_path: Path to the next frame (f002.png)
        autoencoder: Trained MotionAutoencoder instance
        motion_estimator: MotionEstimation instance
        show_results: Whether to display the visualization
    
    Returns:
        warped_frame: The warped frame using decoded motion vectors (clipped to [0,1])
        mse_error: Mean squared error between warped and target frame
    """
    # Load and preprocess frames
    def load_frame(filepath):
        frame = imageio.imread(filepath) / 255.0
        frame = ndimage.zoom(frame, (256/frame.shape[0], 256/frame.shape[1], 1))
        return tf.convert_to_tensor(frame[np.newaxis], dtype=tf.float32)
    
    prev_frame = load_frame(prev_frame_path)
    next_frame = load_frame(next_frame_path)
    
    # Estimate motion between frames
    flow, _ = motion_estimator.estimate_flow(prev_frame, next_frame)
    
    # Encode and decode the motion vectors
    decoded_flow, _ = autoencoder(flow)
    
    # Warp the previous frame using the decoded motion vectors
    warped_frame = tfa.image.dense_image_warp(prev_frame, decoded_flow)
    
    # Clip the warped frame to valid range [0,1]
    clipped_warped = tf.clip_by_value(warped_frame, 0.0, 1.0)
    
    # Calculate error metrics using the clipped version
    mse_error = tf.reduce_mean(tf.square(clipped_warped - next_frame))
    psnr = tf.image.psnr(clipped_warped, next_frame, max_val=1.0)
    
    if show_results:
        # Convert metrics to Python scalars for display
        mse_scalar = mse_error.numpy().item()
        psnr_scalar = psnr.numpy().item()
        
        # Visualize results
        plt.figure(figsize=(18, 6))
        
        # Original Previous Frame
        plt.subplot(1, 4, 1)
        plt.imshow(prev_frame[0].numpy())
        plt.title("Original Previous Frame")
        plt.axis('off')
        
        # Target Next Frame
        plt.subplot(1, 4, 2)
        plt.imshow(next_frame[0].numpy())
        plt.title("Target Next Frame")
        plt.axis('off')
        
        # Warped Frame (clipped)
        plt.subplot(1, 4, 3)
        plt.imshow(clipped_warped[0].numpy())
        plt.title("Warped Frame (Clipped)")
        plt.axis('off')
        
        # Absolute Error
        plt.subplot(1, 4, 4)
        error = tf.abs(clipped_warped - next_frame).numpy()[0]
        error_display = error.mean(axis=-1)  # Convert RGB to grayscale for error display
        plt.imshow(error_display, cmap='hot', vmin=0, vmax=1)
        plt.title(f"Absolute Error\nMSE: {mse_scalar:.4f}\nPSNR: {psnr_scalar:.2f} dB")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return clipped_warped, mse_error