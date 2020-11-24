import logging

from tfor.evaluate import DatasetEvaluator
from .simple_trainer import SimpleTrainer
from ..config import ConfNode
import tensorflow as tf

def build_model(cfg):

    pass


class TFDefaultTrainer(SimpleTrainer):
    def __init__(self, cfg: ConfNode):
        self.cfg = cfg
        data_loader = self.build_train_dataloader(cfg, )
        model = build_model(cfg)
        optimizer = self.build_optimizer(cfg)
        super().__init__(model, data_loader, optimizer)

        self.checkpointer = None

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name='EVAL'):
        pass

    @classmethod
    def build_train_dataloader(cls, cfg, dataset_name='TRAIN'):
        pass

    @classmethod
    def build_evaluator(cls, cfg, dataset_name) -> DatasetEvaluator:
        pass

    def build_optimizer(self, cfg) -> tf.keras.optimizers.Optimizer:
        pass

    def loss_fn(self, inputs_data, outputs):
        pass

    def metric_fn(self, inputs_data, outputs):
        """
        For any non-loss metric for training
        :param inputs_data:
        :param outputs:
        :return:
        """
        pass

    def build_hooks(self, cfg):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        from .hooks import EvalHook

        _hooks = [
        ]

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        _hooks.append(EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        _hooks.append(PeriodicWriter(self.build_writers(), period=20))
        return _hooks

    def build_writers(self):
        pass
    
    



