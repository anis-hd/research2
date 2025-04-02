import tensorflow as tf
import tensorflow_addons as tfa

# Motion Estimation Network
class MotionEstimation(tf.Module):
    def __init__(self, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def _build_pyramid(self, image):
        level3 = tf.keras.layers.AveragePooling2D(2, 2, 'same')(image)
        level2 = tf.keras.layers.AveragePooling2D(2, 2, 'same')(level3)
        level1 = tf.keras.layers.AveragePooling2D(2, 2, 'same')(level2)
        level0 = tf.keras.layers.AveragePooling2D(2, 2, 'same')(level1)
        return [level0, level1, level2, level3, image]

    def _convnet(self, im1_warp, im2, flow, layer):
        x = tf.concat([im1_warp, im2, flow], -1)
        x = tf.keras.layers.Conv2D(32, 7, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, 7, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(32, 7, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(16, 7, padding='same', activation='relu')(x)
        return tf.keras.layers.Conv2D(2, 7, padding='same')(x)

    def _loss(self, flow_coarse, im1, im2, layer):
        flow = tf.image.resize(flow_coarse, [tf.shape(im1)[1], tf.shape(im1)[2]])
        im1_warped = tfa.image.dense_image_warp(im1, flow)
        res = self._convnet(im1_warped, im2, flow, layer)
        flow_fine = res + flow
        im1_warped_fine = tfa.image.dense_image_warp(im1, flow_fine)
        return tf.reduce_mean(tf.square(im1_warped_fine - im2)), flow_fine

    def estimate_flow(self, im1, im2):
        pyramid_im1 = self._build_pyramid(im1)
        pyramid_im2 = self._build_pyramid(im2)
        flow = tf.zeros([self.batch_size, self.height//16, self.width//16, 2])
        losses = []
        for level in range(5):
            loss, flow = self._loss(flow, pyramid_im1[level], pyramid_im2[level], level)
            losses.append(loss)
        return flow, losses