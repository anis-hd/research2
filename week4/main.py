import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from motion_estimation import MotionEstimation
from motion_autoencoder.motion_autoencoder import MotionAutoencoder

def load_and_preprocess_image(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize if needed (optional)
    img = cv2.resize(img, (640, 480))
    return img

def visualize_motion(frame1, frame2, motion_vectors, block_size=16):
    h, w = frame1.shape
    plt.figure(figsize=(15, 5))
    
    # Plot original frames
    plt.subplot(131)
    plt.imshow(frame1, cmap='gray')
    plt.title('Frame 1')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(frame2, cmap='gray')
    plt.title('Frame 2')
    plt.axis('off')
    
    # Plot motion vectors
    plt.subplot(133)
    plt.imshow(frame1, cmap='gray')
    motion_vectors = motion_vectors.reshape(-1, 2)
    for i, (dx, dy) in enumerate(motion_vectors):
        y = (i // (w // block_size)) * block_size + block_size // 2
        x = (i % (w // block_size)) * block_size + block_size // 2
        plt.arrow(x, y, dx, dy, color='r', alpha=0.5)
    plt.title('Motion Vectors')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_autoencoder(model, motion_vectors, num_epochs=10, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            decoded, entropy = model(x, training=True)
            loss = tf.reduce_mean(tf.square(decoded - x)) + 0.01 * tf.reduce_mean(entropy)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(motion_vectors).batch(32).shuffle(1000)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataset:
            loss = train_step(batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def main():
    # Load frames
    frame1 = load_and_preprocess_image('data/f001.png')
    frame2 = load_and_preprocess_image('data/f002.png')
    
    # Convert frames to tensorflow tensors and add batch dimension
    frame1_tf = tf.convert_to_tensor(frame1, dtype=tf.float32)[None, ..., None]
    frame2_tf = tf.convert_to_tensor(frame2, dtype=tf.float32)[None, ..., None]
    
    # Initialize motion estimation
    motion_estimator = MotionEstimation(batch_size=1, height=480, width=640)
    
    # Estimate motion vectors
    motion_vectors, losses = motion_estimator.estimate_flow(frame1_tf, frame2_tf)
    
    # Visualize original motion vectors
    visualize_motion(frame1, frame2, motion_vectors.numpy(), block_size=16)
    
    # Initialize and train autoencoder
    model = MotionAutoencoder(height=480, width=640)
    train_autoencoder(model, motion_vectors, num_epochs=10)
    
    # Visualize compressed motion vectors
    compressed_vectors, _ = model(motion_vectors)
    compressed_vectors = compressed_vectors.numpy()
    
    plt.figure(figsize=(15, 5))
    
    # Original motion vectors
    plt.subplot(131)
    plt.imshow(frame1, cmap='gray')
    motion_vectors_np = motion_vectors.numpy()
    for i, (dx, dy) in enumerate(motion_vectors_np.reshape(-1, 2)):
        y = (i // (frame1.shape[1] // 16)) * 16 + 8
        x = (i % (frame1.shape[1] // 16)) * 16 + 8
        plt.arrow(x, y, dx, dy, color='r', alpha=0.5)
    plt.title('Original Motion Vectors')
    plt.axis('off')
    
    # Compressed motion vectors
    plt.subplot(132)
    plt.imshow(frame1, cmap='gray')
    for i, (dx, dy) in enumerate(compressed_vectors.reshape(-1, 2)):
        y = (i // (frame1.shape[1] // 16)) * 16 + 8
        x = (i % (frame1.shape[1] // 16)) * 16 + 8
        plt.arrow(x, y, dx, dy, color='b', alpha=0.5)
    plt.title('Compressed Motion Vectors')
    plt.axis('off')
    
    # Difference between original and compressed
    plt.subplot(133)
    plt.imshow(frame1, cmap='gray')
    for i, ((dx1, dy1), (dx2, dy2)) in enumerate(zip(motion_vectors_np.reshape(-1, 2), 
                                                    compressed_vectors.reshape(-1, 2))):
        y = (i // (frame1.shape[1] // 16)) * 16 + 8
        x = (i % (frame1.shape[1] // 16)) * 16 + 8
        plt.arrow(x, y, dx1-dx2, dy1-dy2, color='g', alpha=0.5)
    plt.title('Difference (Original - Compressed)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
