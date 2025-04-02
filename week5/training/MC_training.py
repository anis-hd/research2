import tensorflow as tf

def train_mc_network(model, warped_frames, original_frames, motion_vectors, target_frames, epochs=50):
    """
    Train the motion compensation network
    Args:
        model: MC network model
        warped_frames: Tensor of warped frames (B, H, W, 3)
        original_frames: Tensor of original reference frames (B, H, W, 3)
        motion_vectors: Tensor of compressed motion vectors (B, H, W, 2)
        target_frames: Tensor of target frames to predict (B, H, W, 3)
        epochs: Number of training epochs
    """
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Combine inputs: concatenate warped frame + original frame + motion vectors
    inputs = tf.concat([warped_frames, original_frames, motion_vectors], axis=-1)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs, target_frames))
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, (inputs, targets) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
                
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(batch+1):.4f}")
    
    return model