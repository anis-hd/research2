"""import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

# Import from our custom modules
from models.motion_estimation import MotionEstimation
from models.autoencoder import MotionAutoencoder
from models.entropy_model import EntropyModel
from models.mc_network import build_MC_model
from utils.visualization import visualize_results, visualize_in_notebook
from utils.warping import warp_frame_with_motion_vectors
from training.train_autoencoder import train_autoencoder

def main(frame1_path, frame2_path):
    # 1. Load and preprocess frames
    def load_frame(path):
        frame = imageio.imread(path) / 255.0
        frame = ndimage.zoom(frame, (256/frame.shape[0], 256/frame.shape[1], 1))
        return tf.convert_to_tensor(frame[np.newaxis], dtype=tf.float32)
    
    print("Loading frames...")
    im1 = load_frame(frame1_path)
    im2 = load_frame(frame2_path)

    # 2. Motion Estimation
    print("\nEstimating motion between frames...")
    motion_estimator = MotionEstimation(batch_size=1, height=256, width=256)
    flow, _ = motion_estimator.estimate_flow(im1, im2)
    flow_np = flow.numpy()

    # 3. Train and Save Autoencoder
    print("\nTraining motion vector autoencoder...")
    autoencoder = MotionAutoencoder(256, 256)
    losses, rates, dists, q_steps = train_autoencoder(autoencoder, flow_np, epochs=200)
    autoencoder.save_weights('models/autoencoder_weights.h5')
    
    # Visualize autoencoder training results
    decoded, entropy = autoencoder(flow)
    visualize_results(flow_np, decoded.numpy(), entropy, losses, dists, rates, q_steps)
    visualize_in_notebook(flow_np, decoded.numpy(), entropy, losses, dists, rates, q_steps)

    # 4. Warp with Compressed Vectors
    print("\nWarping frame with compressed motion vectors...")
    warped_frame, warp_error = warp_frame_with_motion_vectors(
        frame1_path, 
        frame2_path, 
        autoencoder, 
        motion_estimator,
        show_results=True
    )

    # 5. Train and Save MC Network
    print("\nTraining Motion Compensation network...")
    mc_model = build_MC_model((256, 256, 3))
    mc_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    mc_model.fit(warped_frame, im2, epochs=50, batch_size=1)
    mc_model.save_weights('models/mc_model_weights.h5')

    # 6. Generate Final Compensation
    print("\nGenerating motion compensated frame...")
    compensated_frame = mc_model(warped_frame)
    compensation_error = tf.reduce_mean(tf.square(compensated_frame - im2))
    print(f"Final compensation error (MSE): {compensation_error:.4f}")

    # 7. Comprehensive Visualization
    print("\nGenerating results visualization...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original Flow
    orig_mag = np.sqrt(flow_np[0,...,0]**2 + flow_np[0,...,1]**2)
    axs[0,0].imshow(orig_mag, cmap='jet')
    axs[0,0].set_title('Original Motion Vectors')

    # Compressed Flow
    decoded_flow, _ = autoencoder(flow)
    comp_mag = np.sqrt(decoded_flow.numpy()[0,...,0]**2 + decoded_flow.numpy()[0,...,1]**2)
    axs[0,1].imshow(comp_mag, cmap='jet')
    axs[0,1].set_title('Compressed Motion Vectors')

    # Warped Frame
    axs[0,2].imshow(warped_frame.numpy().squeeze())
    axs[0,2].set_title(f'Warped Frame (MSE: {warp_error:.2f})')

    # Compensated Frame
    axs[1,0].imshow(compensated_frame.numpy().squeeze())
    axs[1,0].set_title('Motion Compensated Frame')

    # Target Frame
    axs[1,1].imshow(im2.numpy().squeeze())
    axs[1,1].set_title('Target Frame')

    # Error Map
    error = np.abs(compensated_frame.numpy() - im2.numpy()).squeeze()
    axs[1,2].imshow(error.mean(axis=-1), cmap='hot')
    axs[1,2].set_title(f'Error (MSE: {compensation_error:.2f})')

    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nProcessing complete. Results saved to:")
    print("- models/autoencoder_weights.h5")
    print("- models/mc_model_weights.h5")
    print("- results/comparison_results.png")

if __name__ == "__main__":
    main(
        'data/f001.png',
        'data/f002.png'
    )

    """


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
from training.MC_training import train_mc_network
from models.mc_network import build_MC_model
from models.mc_network import MC_network
from models.mc_network import resblock

from models.motion_estimation import MotionEstimation
from models.autoencoder import MotionAutoencoder
from training.train_autoencoder import train_autoencoder  # Changed this import

from utils.visualization import visualize_results, visualize_in_notebook
from utils.warping import warp_frame_with_motion_vectors

def main():
    # Initialize models
    BATCH_SIZE = 1
    HEIGHT, WIDTH = 256, 256
    
    print("Initializing models...")
    motion_estimator = MotionEstimation(BATCH_SIZE, HEIGHT, WIDTH)
    autoencoder = MotionAutoencoder(HEIGHT, WIDTH)

    # Load sample frames
    print("\nLoading frames...")
    prev_path = r'C:\Users\anish\OneDrive\Desktop\pfe\research2\week5\data\f001.png'
    next_path = r'C:\Users\anish\OneDrive\Desktop\pfe\research2\week5\data\f002.png'

    # Estimate initial flow
    print("\nEstimating initial motion...")
    im1 = imageio.imread(prev_path) / 255.0
    im2 = imageio.imread(next_path) / 255.0
    im1 = ndimage.zoom(im1, (HEIGHT/im1.shape[0], WIDTH/im1.shape[1], 1))
    im2 = ndimage.zoom(im2, (HEIGHT/im2.shape[0], WIDTH/im2.shape[1], 1))
    im1_tensor = tf.convert_to_tensor(im1[np.newaxis], dtype=tf.float32)
    im2_tensor = tf.convert_to_tensor(im2[np.newaxis], dtype=tf.float32)
    
    flow, _ = motion_estimator.estimate_flow(im1_tensor, im2_tensor)

    # Train autoencoder
    print("\nTraining autoencoder...")
    losses, rates, dists, q_steps = train_autoencoder(autoencoder, flow.numpy(), epochs=50)
    # Visualize results
    print("\nVisualizing results...")
    decoded_flow, entropy = autoencoder(flow)
    visualize_results(
        flow.numpy(), 
        decoded_flow.numpy(), 
        entropy,
        losses, 
        dists, 
        rates, 
        q_steps
    )
    # Perform warping
    print("\nPerforming warping...")
    warped_frame, mse = warp_frame_with_motion_vectors(
        prev_path,
        next_path,
        autoencoder,
        motion_estimator
    )

    #implement motion compensation here


    # Prepare MC Network inputs
    print("\nPreparing MC Network inputs...")
    # Load original reference frame
    
    # Define load_frame function
    def load_frame(path):
        frame = imageio.imread(path) / 255.0
        frame = ndimage.zoom(frame, (256/frame.shape[0], 256/frame.shape[1], 1))
        return tf.convert_to_tensor(frame[np.newaxis], dtype=tf.float32)

    # Get required components
    ref_frame = load_frame(prev_path)  # Original reference frame
    decoded_flow, _ = autoencoder(flow)  # Compressed motion vectors
    
    # Build and train MC Network
    mc_model = build_MC_model((256, 256, 8))
    
    print("\nTraining Motion Compensation Network...")
    mc_model = train_mc_network(
        mc_model,
        warped_frame,
        ref_frame,
        decoded_flow,
        im2_tensor,  # Target frame
        epochs=50
    )
    
    # Generate final compensation
    print("\nGenerating motion compensated frame...")
    combined_input = tf.concat([warped_frame, ref_frame, decoded_flow], axis=-1)
    compensated_frame = mc_model(combined_input)





    # Calculate compensation error
    compensation_error = tf.reduce_mean(tf.square(compensated_frame - im2_tensor))
    print(f"Compensation MSE: {compensation_error.numpy():.4f}")
    residual = compensated_frame - im1_tensor
    
    # Add residual visualization to existing plot
    plt.figure(figsize=(18, 15))  # Increase figure size
    # Visualization
    print("\nVisualizing results...")
    plt.figure(figsize=(18, 12))
    
    # Original Frames
    plt.subplot(2, 3, 1)
    plt.imshow(im1_tensor.numpy().squeeze())
    plt.title("Original Frame (f001)")
    
    plt.subplot(2, 3, 2)
    plt.imshow(im2_tensor.numpy().squeeze())
    plt.title("Target Frame (f002)")
    
    # Warped Frame
    plt.subplot(2, 3, 3)
    plt.imshow(warped_frame.numpy().squeeze().clip(0, 1))
    plt.title(f"Warped Frame (MSE: {mse.numpy():.2f})")
    
    # Compensated Frame
    plt.subplot(2, 3, 4)
    plt.imshow(compensated_frame.numpy().squeeze().clip(0, 1))
    plt.title(f"Compensated Frame (MSE: {compensation_error.numpy():.2f})")
    
    # Error Maps
    plt.subplot(2, 3, 5)
    plt.imshow(np.abs(warped_frame.numpy() - im2_tensor.numpy()).squeeze().mean(-1), cmap='hot')
    plt.title("Warping Error")
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.abs(compensated_frame.numpy() - im2_tensor.numpy()).squeeze().mean(-1), cmap='hot')
    plt.title("Compensation Error")
    
    plt.tight_layout()
    plt.savefig('motion_compensation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

  # Residual Visualization
    plt.subplot(3, 3, 7)
    residual_mag = np.abs(residual.numpy().squeeze())
    plt.imshow(residual_mag.mean(axis=-1), cmap='hot', vmax=np.percentile(residual_mag, 95))
    plt.title("Residual Magnitude (Compensated - f001)")
    plt.colorbar()
    
    # Residual Histogram
    plt.subplot(3, 3, 8)
    plt.hist(residual.numpy().flatten(), bins=100, color='purple')
    plt.title("Residual Distribution")
    plt.xlabel("Pixel Value Difference")
    
    # Residual Vector Visualization (for motion analysis)
    plt.subplot(3, 3, 9)
    step = 16
    X, Y = np.meshgrid(np.arange(0, 256, step), np.arange(0, 256, step))
    plt.quiver(X, Y, 
               residual.numpy()[0, ::step, ::step, 0],  # Horizontal component
               residual.numpy()[0, ::step, ::step, 1],   # Vertical component
               scale=50, color='white')
    plt.title("Residual Motion Components")
    
    plt.tight_layout()
    plt.savefig('motion_compensation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()