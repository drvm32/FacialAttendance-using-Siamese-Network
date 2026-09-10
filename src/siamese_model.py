import tensorflow as tf
from tensorflow.keras import layers, Model

def create_base_network(input_shape=(128, 128, 3)):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation=None)
    ])
    return model

def siamese_network(input_shape):
    base_network = create_base_network(input_shape)
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    output_a = base_network(input_a)
    output_b = base_network(input_b)
    distance = layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1)),
        output_shape=(None,)
    )([output_a, output_b])
    distance = layers.Reshape((-1, 1))(distance)
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model, base_network

if __name__ == "__main__":
    model, base_network = siamese_network((128, 128, 3))
    model.summary()