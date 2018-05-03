import tensorflow as tf
import numpy as np
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator

if __name__ == "__main__":
    generator = ImageDataGenerator('trainlist.txt', 'training', 6, 10, shuffle=True)
    data = generator.data
    iterator = Iterator.from_structure(generator.data.output_types, generator.data.output_shapes)
    next_batch = iterator.get_next()
    init = iterator.make_initializer(generator.data)
    with tf.Session() as sess:
        sess.run(init)
        for step in range(20):
            print("step:", step)
            img_batch, label_batch = sess.run(next_batch)
            print(np.argmax(label_batch, axis=1))
            # generator.reshuffle_data()
