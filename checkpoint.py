import numpy as np
import tensorflow as tf
from alexnet import AlexNet, SiameseAlexNet, Model
from tensorflow.python.client.session import BaseSession


class Checkpointer:
    def __init__(self, name: str, model: Model, save_path, higher_is_better=True, sess=None):
        self.best = -1e10 if higher_is_better else 1e10
        self.best_type = "Highest" if higher_is_better else "Lowest"
        self.name = name
        self.model = model
        self.save_path = save_path
        self.higher_is_better = higher_is_better
        self.session = None
        if sess is not None:
            self.update_session(sess)

    def update_session(self, sess: BaseSession):
        assert isinstance(sess, BaseSession), "sess is not a TensorFlow Session"
        self.session = sess

    def update_best(self, value, checkpoint=True):
        if self._better_than_best(value):
            self._update_new_best(value, checkpoint=checkpoint)
        else:
            self._retain_current_best()

    def _better_than_best(self, value) -> bool:
        # implement a reader-friendly xor function
        if self.higher_is_better:
            return value > self.best
        else:
            return value < self.best

    def _update_new_best(self, new_best, checkpoint=True):
        assert self.session is not None
        print(self.best_type, self.name, "updated {} ---> {}".format(self.best, new_best))
        self.best = new_best

        if checkpoint:
            print("Saving checkpoint at", self.save_path)
            self.model.save_model_vars(self.save_path, self.session)
            print("Checkpint Saved")
        else:
            print("Not Saving checkpoint due to configuration")

    def _retain_current_best(self):
        print(self.best_type, self.name, "remained {}".format(self.best))


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], name="x")
    keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")
    save_path = "/Users/liushuheng/Desktop/vars"
    name = "xent"
    net = AlexNet(x, keep_prob, 3, ['fc8'])
    checkpointer = Checkpointer(name, net, save_path, higher_is_better=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(20):
            checkpointer.update_session(sess)
            new_value = np.random.rand(1)
            print("\nnew value = {}".format(new_value))
            checkpointer.update_best(new_value, checkpoint=False)

    print(checkpointer.best)

