"""Python Script Template."""
import numpy as np
from gym.utils import colorize

from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model import ExactGPModel, TransformedModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.training.model_learning import (
    calibrate_model,
    evaluate_model,
    train_model,
)


class ModelLearningAlgorithm(object):
    """An algorithm for model learning.

    Parameters
    ----------
    model_optimizer: Optimizer, optional.
        Optimizer to learn parameters of model.
    num_epochs: int.
        Number of epochs to iterate through the dataset.
    batch_size: int.
        Batch size of optimization algorithm.
    bootstrap: bool.
        Flag that indicates whether to add bootstrapping to dataset.
    max_memory: int.
        Maximum size of dataset.
    validation_ratio: float.
        Validation set ratio.


    Methods
    -------
    update_model_posterior(self, last_trajectory: Trajectory, logger: Logger) -> None:
        Update model posterior of GP models.
    learn(self, last_trajectory: Trajectory, logger: Logger) -> None: ...
        Learn using stochastic gradient descent on marginal maximum likelihood.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        termination_model=None,
        model_optimizer=None,
        num_epochs=1,
        batch_size=100,
        bootstrap=True,
        max_memory=10000,
        epsilon=0.1,
        non_decrease_iter=5,
        validation_ratio=0.1,
        calibrate=True,
        num_memory_steps=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        if not isinstance(reward_model, TransformedModel):
            reward_model = TransformedModel(reward_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model

        assert self.dynamical_model.model_kind == "dynamics"
        assert self.reward_model.model_kind == "rewards"
        if self.termination_model is not None:
            assert self.termination_model.model_kind == "termination"

        self.model_optimizer = model_optimizer

        if hasattr(self.dynamical_model.base_model, "num_heads"):
            num_heads = self.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        # Note: The transformations are shared by both data sets.
        self.train_set = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
            num_memory_steps=num_memory_steps,
        )
        self.validation_set = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
            num_memory_steps=num_memory_steps,
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.non_decrease_iter = non_decrease_iter
        self.validation_ratio = validation_ratio
        self.calibrate = calibrate

        if self.num_epochs > 0:
            assert self.model_optimizer is not None

    def _update_model_posterior(self, last_trajectory):
        """Update model posterior of GP-models with new data."""
        if isinstance(self.dynamical_model.base_model, ExactGPModel):
            observation = stack_list_of_tuples(last_trajectory)  # Parallelize.
            if observation.action.shape[-1] > self.dynamical_model.dim_action[0]:
                observation.action = observation.action[
                    ..., : self.dynamical_model.dim_action[0]
                ]
            for transform in self.train_set.transformations:
                observation = transform(observation)
            print(colorize("Add data to GP Model", "yellow"))
            self.dynamical_model.base_model.add_data(
                observation.state, observation.action, observation.next_state
            )

            print(colorize("Summarize GP Model", "yellow"))
            self.dynamical_model.base_model.summarize_gp()

    def add_last_trajectory(self, last_trajectory):
        """Add last trajectory to learning algorithm."""
        self._update_model_posterior(last_trajectory)
        for observation in last_trajectory:
            observation = observation.clone()
            if observation.action.shape[-1] > self.dynamical_model.dim_action[0]:
                observation.action = observation.action[
                    ..., : self.dynamical_model.dim_action[0]
                ]  # Only get real actions.
            if np.random.rand() < self.validation_ratio:
                self.validation_set.append(observation)
            else:
                self.train_set.append(observation)

    def _learn(
        self, model, logger, calibrate=False, max_iter=None, dynamical_model=None
    ):
        """Learn a model."""
        print(colorize(f"Training {model.model_kind} model", "yellow"))
        num_epochs = self.num_epochs if max_iter is None else None
        train_model(
            model=model,
            train_set=self.train_set,
            validation_set=self.validation_set,
            batch_size=self.batch_size,
            max_iter=max_iter,
            num_epochs=num_epochs,
            optimizer=self.model_optimizer,
            logger=logger,
            epsilon=self.epsilon,
            non_decrease_iter=self.non_decrease_iter,
            dynamical_model=dynamical_model,
        )
        if (
            calibrate
            and not model.deterministic
            and len(self.validation_set) > self.batch_size
        ):
            calibrate_model(model, self.validation_set, self.num_epochs, logger=logger)

    def learn(self, logger, max_iter=None):
        """Learn using stochastic gradient descent on marginal maximum likelihood."""
        self._learn(
            self.dynamical_model.base_model,
            logger,
            calibrate=self.calibrate,
            max_iter=max_iter,
            dynamical_model=self.dynamical_model.base_model,
        )
        if len(self.validation_set) > self.batch_size:
            validation_data = self.validation_set.all_raw
            evaluate_model(
                self.dynamical_model,
                validation_data,
                logger,
                dynamical_model=self.dynamical_model,
            )

        if any(p.requires_grad for p in self.reward_model.parameters()):
            self._learn(
                self.reward_model.base_model,
                logger,
                calibrate=self.calibrate,
                max_iter=max_iter,
                dynamical_model=self.dynamical_model.base_model,
            )
            if len(self.validation_set) > self.batch_size:
                evaluate_model(
                    self.reward_model,
                    validation_data,
                    logger,
                    dynamical_model=self.dynamical_model,
                )

        if self.termination_model is not None and any(
            p.requires_grad for p in self.termination_model.parameters()
        ):
            self._learn(
                self.termination_model,
                logger,
                calibrate=False,
                max_iter=max_iter,
                dynamical_model=self.dynamical_model.base_model,
            )
            if len(self.validation_set) > self.batch_size:
                evaluate_model(
                    self.termination_model,
                    validation_data,
                    logger,
                    dynamical_model=self.dynamical_model,
                )

        if isinstance(self.dynamical_model.base_model, ExactGPModel):
            for i, gp in enumerate(self.dynamical_model.base_model.gp):
                logger.update(**{f"gp{i} num inputs": len(gp.train_targets)})

                if isinstance(gp, SparseGP):
                    logger.update(**{f"gp{i} num inducing inputs": gp.xu.shape[0]})
