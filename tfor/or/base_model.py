import tensorflow as tf


class TFORModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=False, **kwargs):
        pass

    def production_saved_model(self):
        print()

