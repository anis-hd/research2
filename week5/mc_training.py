import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.motion_estimation import MotionEstimation
from models.autoencoder import MotionAutoencoder
from models.mc_network import build_MC_model
from utils.warping import warp_frame_with_motion_vectors
from utils.visualization import visualize_results

def train_mc_model(frame1_path, frame2_path):
    # 1. Load frames
    def load_frame(path):
        frame = imageio.imread(path) / 255.0
        frame = ndimage.zoom(frame, (256/frame.shape[0], 256/frame.shape[1], 1))
        return tf.convert_to_tensor(frame[np.newaxis], dtype=tf.float32)
    
    frame1 = load_frame(frame1_path)
    frame2 = load_frame(frame2_path)

    # 2. Get warped frame using compressed motion vectors
    motion_estimator = MotionEstimation(1, 256, 256)
    autoencoder = MotionAutoencoder(256, 256)
    
    # Load pretrained autoencoder weights
    autoencoder.load_weights("models/autoencoder_weights.h5")  # Assuming pre-trained
    
    # Get warped frame
    warped_frame, _ = warp_frame_with_motion_vectors(
        frame1_path,
        frame2_path,
        autoencoder,
        motion_estimator,
        show_results=False
    )

    # 3. Build and train MC network
    mc_model = build_MC_model((256, 256, 3))
    mc_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss='mse',
                    metrics=['mae', tf.keras.metrics.PSNR])
    
    print("Training MC network...")
    history = mc_model.fit(
        warped_frame,
        frame2,
        epochs=100,
        batch_size=1,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('models/mc_weights.h5', save_best_only=True)
        ]
    )

    # 4. Generate final compensated frame
    compensated_frame = mc_model.predict(warped_frame)
    
    # 5. Visualize results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(warped_frame.numpy().squeeze())
    plt.title("Input Warped Frame")
    
    plt.subplot(1, 3, 2)
    plt.imshow(compensated_frame.squeeze())
    plt.title("MC Compensated Frame")
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(compensated_frame.squeeze() - frame2.numpy().squeeze()).mean(axis=-1), cmap='hot')
    plt.title("Compensation Error")
    plt.colorbar()
    
    plt.savefig('results/mc_results.png')
    plt.show()

    # Plot training metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr'], label='Train PSNR')
    plt.plot(history.history['val_psnr'], label='Val PSNR')
    plt.title('Quality Metrics')
    plt.legend()
    
    plt.savefig('results/mc_training_curves.png')
    plt.show()

if __name__ == "__main__":
    train_mc_model(
        'data/f001.png',
        'data/f002.png'
    )