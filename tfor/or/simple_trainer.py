import logging
from collections import OrderedDict

from ..evaluate import DatasetEvaluator, inference_on_dataset
from .base_trainer import TrainerBase
import time
from typing import Dict, Any
import tensorflow as tf
import numpy as np


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def loss_fn(self, inputs_data, outputs) -> dict:
        return {}

    def metric_fn(self, inputs_data, outputs) -> dict:
        return {}

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        inputs_data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with tf.GradientTape() as tape:
            outputs = self.model(inputs_data, training=True)
            loss_dict = self.loss_fn(inputs_data, outputs)
            metric_dict = self.metric_fn(inputs_data, outputs)

        loss_value = sum(loss_dict.values())
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # TODO: Do gradient clipping or accumulation here
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_and_metric_dict = dict(**loss_dict, **metric_dict)
        self._write_training_metrics(loss_and_metric_dict, data_time)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            print(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_train_dataloader(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    def _write_training_metrics(self, loss_dict: Dict, data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward

        metrics_dict = {k: float(v) for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = self.storage

        # data_time among workers can have high variance. The actual latency
        # caused by data_time is the maximum among workers.
        data_time = np.max([x.pop("data_time") for x in metrics_dict])
        storage.put_scalar("data_time", data_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in metrics_dict]) for k in metrics_dict[0].keys()
        }
        total_losses_reduced = sum(metrics_dict.values())
        if not np.isfinite(total_losses_reduced):
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={self.iter}!\n"
                f"loss_dict = {metrics_dict}"
            )

        storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            storage.put_scalars(**metrics_dict)
