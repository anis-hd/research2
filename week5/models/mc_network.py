import tensorflow as tf

# -------------------------------------------------------------------------------
def resblock(input_tensor, IC, OC, name):
    with tf.name_scope(name):
        l1 = tf.nn.relu(input_tensor, name=name + '_relu1')
        l1 = tf.keras.layers.Conv2D(
            filters=min(IC, OC),
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name=name + '_l1'
        )(l1)
        l2 = tf.nn.relu(l1, name=name + '_relu2')
        l2 = tf.keras.layers.Conv2D(
            filters=OC,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name=name + '_l2'
        )(l2)
        if IC != OC:
            input_tensor = tf.keras.layers.Conv2D(
                filters=OC,
                kernel_size=1,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                name=name + '_map'
            )(input_tensor)
        return input_tensor + l2

def MC_network(input_tensor):
    # m1
    m1 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=1, padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='mc1'
    )(input_tensor)
    # m2
    m2 = resblock(m1, 64, 64, name='mc2')
    # m3: average pooling
    m3 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(m2)
    # m4
    m4 = resblock(m3, 64, 64, name='mc4')
    # m5: average pooling
    m5 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(m4)
    # m6
    m6 = resblock(m5, 64, 64, name='mc6')
    # m7
    m7 = resblock(m6, 64, 64, name='mc7')
    # m8: upsample m7 by factor 2 and add m4
    m7_shape = tf.shape(m7)
    new_height = 2 * m7_shape[1]
    new_width = 2 * m7_shape[2]
    m8 = tf.image.resize(m7, size=[new_height, new_width], method='bilinear', name='resize_m8')
    m8 = tf.add(m4, m8, name='add_m8')
    # m9
    m9 = resblock(m8, 64, 64, name='mc9')
    # m10: upsample m9 by factor 2 and add m2
    m9_shape = tf.shape(m9)
    new_height = 2 * m9_shape[1]
    new_width = 2 * m9_shape[2]
    m10 = tf.image.resize(m9, size=[new_height, new_width], method='bilinear', name='resize_m10')
    m10 = tf.add(m2, m10, name='add_m10')
    # m11
    m11 = resblock(m10, 64, 64, name='mc11')
    # m12
    m12 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=1, padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='mc12'
    )(m11)
    m12 = tf.nn.relu(m12, name='relu12')
    # m13: final output layer (RGB image)
    m13 = tf.keras.layers.Conv2D(
        filters=3, kernel_size=3, strides=1, padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='mc13'
    )(m12)
    return m13

def build_MC_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = MC_network(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='MC_model')
    return model