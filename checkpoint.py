import numpy as np
import tensorflow as tf
import heapq
from alexnet import AlexNet, SiameseAlexNet, Model
from tensorflow.python.client.session import BaseSession


class MemCache:
    def __init__(self, parameters, index, metric):
        self._parameters = parameters
        self._index = index
        self._metric = metric

    def get_parameters(self):
        return self._parameters

    def get_index(self):
        return self._index

    def get_metric(self):
        return self._metric

    def set_parameters(self, parameters):
        self._parameters = parameters

    def set_index(self, index):
        self._index = index

    def sef_metric(self, metric):
        self._metric = metric


class Checkpointer:
    def __init__(self, name: str, model: Model, save_path, higher_is_better=True, sess=None, mem_size=5):
        self.best = -1e10 if higher_is_better else 1e10
        self.best_type = "Highest" if higher_is_better else "Lowest"
        self.name = name
        self.model = model
        self.save_path = save_path
        self.higher_is_better = higher_is_better
        self.session = None
        # initiate self._mem_caches with the default set of values
        self._mem_caches = [MemCache(model.get_model_vars(sess), -1, -1e10 if higher_is_better else 1e10)]
        self._mem_size = mem_size
        self.heaper_func = heapq.nlargest if higher_is_better else heapq.nsmallest
        if sess is not None:
            self.update_session(sess)

    def update_session(self, sess: BaseSession):
        assert isinstance(sess, BaseSession), "sess is not a TensorFlow Session"
        self.session = sess

    def add_memory_cache(self, mem_cache):
        if not isinstance(mem_cache, MemCache):
            print("Ignoring non-MemCache instance", mem_cache)
            return
        self._mem_caches.append(mem_cache)
        self._mem_caches = self.heaper_func(self._mem_size, self._mem_caches, key=lambda cache: cache.get_metric())

    def list_memory_caches(self):
        return self._mem_caches

    def update_best(self, value, epoch=0, checkpoint=True, mem_cache=False):
        if self._better_than_current_best(value):
            self._update_new_best(value, checkpoint=checkpoint)
        else:
            self._retain_current_best()

        if mem_cache:
            try:
                parameters = self.model.get_model_vars(self.session)
                self.add_memory_cache(MemCache(parameters, epoch, value))
            except AttributeError as e:
                print(e)
                print("Default: not updating memory cache")

    def _better_than_current_best(self, value) -> bool:
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
            print("Checkpoint Saved")
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
        for epoch in range(20):
            checkpointer.update_session(sess)
            new_value = np.random.rand(1)
            print("\nnew value = {}".format(new_value))
            checkpointer.update_best(new_value, epoch=epoch, checkpoint=False, mem_cache=True)

    print(checkpointer.best)
    mem_caches = checkpointer.list_memory_caches()
    # print(checkpointer.list_memory_caches())
