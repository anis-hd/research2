import tensorflow as tf

def train_autoencoder(autoencoder, flow, epochs=50, beta=0.01):
    optimizer = tf.keras.optimizers.Adam(0.001)
    flow = tf.convert_to_tensor(flow, dtype=tf.float32)

    # Calculate magnitude for normalization
    flow_magnitude = tf.norm(flow, axis=-1)
    max_magnitude = tf.reduce_max(flow_magnitude)

    losses, rates, dists, q_steps = [], [], [], []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            decoded, entropy = autoencoder(flow, training=True)

            # Magnitude-based distortion
            orig_magnitude = tf.norm(flow, axis=-1)
            recon_magnitude = tf.norm(decoded, axis=-1)
            distortion = tf.reduce_mean(tf.abs(orig_magnitude - recon_magnitude))

            rate = tf.reduce_mean(entropy)
            loss = distortion + beta * rate

        grads = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))

        losses.append(loss.numpy())
        rates.append(rate.numpy())
        dists.append(distortion.numpy())
        q_steps.append(autoencoder.quantization_step.numpy())

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Distortion: {distortion:.4f}")
            print(f"  Rate: {rate:.4f}")
            print(f"  Quant Step: {autoencoder.quantization_step.numpy():.4f}")
            print(f"  Decoded Range: [{tf.reduce_min(decoded):.4f}, {tf.reduce_max(decoded):.4f}]")
            print(f"  Flow Range: [{tf.reduce_min(flow):.4f}, {tf.reduce_max(flow):.4f}]")

    return losses, rates, dists, q_steps
