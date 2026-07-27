"""
Learning process for training and evaluating models.
"""

from __future__ import annotations

from typing import cast, Protocol

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree, Array, PRNGKeyArray
from tqdm.auto import tqdm

Model = TypeVar("Model", bound=eqx.Module)
Batch = TypeVar("Batch")


@dataclass
class LearningState(Generic[Model], eqx.Module):
    model_trainable: Model
    model_static: Model
    optimizer_state: optax.OptState
    optimizer: optax.GradientTransformation = field(repr=False)

    @classmethod
    def create(
        cls,
        model: Model,
        optimizer: optax.GradientTransformation,
        trainable_mask: PyTree | None = None,
    ) -> "LearningState[Model]":
        if trainable_mask is None:
            trainable_mask = jax.tree_util.tree_map(
                eqx.is_inexact_array, model
            )
        model_trainable, model_static = eqx.partition(model, trainable_mask)
        optimizer_state = optimizer.init(model_trainable)
        return cls(model_trainable, model_static, optimizer_state, optimizer)

    @property
    def model(self) -> Model:
        return eqx.combine(self.model_trainable, self.model_static)


def split_random_key(
    random_key: PRNGKeyArray | None,
) -> tuple[PRNGKeyArray | None, PRNGKeyArray | None]:
    if random_key is not None:
        new_key, remaining_key = jax.random.split(random_key)
        return new_key, remaining_key
    else:
        return None, None


ModelInputT = TypeVar("ModelInputT", bound=eqx.Module, contravariant=True)


# Define loss function type hint
class LossFunctionProtocol(Protocol[ModelInputT]):
    def __call__(
        self, model: ModelInputT, batch: Any, random_key: PRNGKeyArray | None
    ) -> Array: ...


def make_train_step(
    loss_fn: LossFunctionProtocol[Model],
) -> Callable[
    [LearningState[Model], Batch, PRNGKeyArray | None],
    tuple[LearningState[Model], Array],
]:
    @eqx.filter_jit
    def train_step(
        learning_state: LearningState[Model],
        batch: Batch,
        random_key: PRNGKeyArray | None,
    ) -> tuple[LearningState[Model], Array]:
        def partitioned_loss(model_trainable: Model) -> Array:
            model = eqx.combine(model_trainable, learning_state.model_static)
            return loss_fn(model, batch, random_key)

        loss, grads = eqx.filter_value_and_grad(partitioned_loss)(
            learning_state.model_trainable
        )

        updates, optimizer_state = learning_state.optimizer.update(
            grads,
            learning_state.optimizer_state,
            cast(optax.Params, learning_state.model_trainable),
        )
        model_trainable = eqx.apply_updates(
            learning_state.model_trainable, updates
        )

        learning_state = LearningState[Model](
            model_trainable=model_trainable,
            model_static=learning_state.model_static,
            optimizer_state=optimizer_state,
            optimizer=learning_state.optimizer,
        )

        return learning_state, loss

    return train_step


def make_evaluate_step(
    loss_fn: LossFunctionProtocol[Model],
) -> Callable[[Model, Batch, PRNGKeyArray | None], Array]:
    @eqx.filter_jit
    def eval_step(
        model: Model, batch: Batch, random_key: PRNGKeyArray | None
    ) -> jax.Array:
        return loss_fn(model, batch, random_key)

    return eval_step


@dataclass
class LearningProcessInfo:
    training_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    epoch: int = 0
    iteration: int = 0


class LearningProcess(Generic[Model]):
    def __init__(
        self,
        model: Model,
        loss_function: LossFunctionProtocol[Model],
        optimizer: optax.GradientTransformation,
        trainable_mask: PyTree | None = None,
    ) -> None:
        self.learning_state = LearningState[Model].create(
            model, optimizer, trainable_mask
        )
        self._train_step = make_train_step(loss_function)
        self._evaluate_step = make_evaluate_step(loss_function)
        self._info = LearningProcessInfo()

    @property
    def model(self) -> Model:
        return self.learning_state.model

    def evaluate_model(
        self,
        data_loader: Iterable[Any],
        random_key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        losses = jnp.array(
            [
                self._evaluate_step(self.model, batch, random_key)
                for batch in data_loader
            ]
        )
        return losses

    def train_one_epoch(
        self,
        training_data_loader: Iterable[Any],
        validation_data_loader: Iterable[Any] | None = None,
        random_key: PRNGKeyArray | None = None,
        validate_every: int | None = None,
        progress_bar: bool = True,
    ) -> None:
        batch_loader = (
            tqdm(training_data_loader, desc=f"epoch {self._info.epoch}")
            if progress_bar
            else training_data_loader
        )

        for batch in batch_loader:
            training_random_key, random_key = split_random_key(random_key)

            self.learning_state, loss = self._train_step(
                self.learning_state, batch, training_random_key
            )
            loss_value = float(loss)
            self._info.training_losses.append(loss_value)
            self._info.iteration += 1

            if progress_bar:
                batch_loader.set_postfix(loss=loss_value)  # type: ignore[union-attr]

            if (
                validation_data_loader is not None
                and validate_every is not None
                and self._info.iteration % validate_every == 0
            ):
                validation_random_key, random_key = split_random_key(
                    random_key
                )
                validation_losses = self.evaluate_model(
                    validation_data_loader, validation_random_key
                )
                self._info.validation_losses.append(
                    float(jnp.mean(validation_losses))
                )

        self._info.epoch += 1

    def train_model(
        self,
        training_data_loader_factory: Callable[[], Iterable[Any]],
        validation_data_loader: Iterable[Any] | None = None,
        random_key: PRNGKeyArray | None = None,
        num_epochs: int = 1,
        validate_every: int | None = None,
        progress_bar: bool = True,
    ) -> LearningProcessInfo:
        if validation_data_loader is not None and self._info.epoch == 0:
            initial_validation_random_key, random_key = split_random_key(
                random_key
            )
            initial_losses = self.evaluate_model(
                validation_data_loader, initial_validation_random_key
            )
            self._info.validation_losses.append(
                float(jnp.mean(initial_losses))
            )

        for _ in range(num_epochs):
            epoch_random_key, random_key = split_random_key(random_key)
            self.train_one_epoch(
                training_data_loader_factory(),
                validation_data_loader,
                epoch_random_key,
                validate_every,
                progress_bar,
            )

        return self._info
