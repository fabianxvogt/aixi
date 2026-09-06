"""Bounded counterexample for prediction-versus-control sufficiency."""

import numpy as np
import pytest
import torch
from torch import nn

from src.actor import BinaryAdvantageHead
from src.observer.models import Observer
from src.utils.config import Transition


def _exact_two_state_observer() -> Observer:
    observer = Observer(
        obs_dim=1,
        action_dim=2,
        latent_dim=1,
        encoder_hidden=[1],
        predictor_hidden=[1],
    )
    with torch.no_grad():
        for parameter in observer.parameters():
            parameter.zero_()
        for layer in observer.encoder.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(1.0)
    return observer


def _fit_binary_actor(
    latents: torch.Tensor,
    rewards: torch.Tensor,
) -> BinaryAdvantageHead:
    actor = BinaryAdvantageHead(latent_dim=1)
    preference_signs = torch.where(
        rewards[:, 1] > rewards[:, 0],
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    design = torch.cat([latents, torch.ones_like(latents)], dim=1)
    weight, bias = torch.linalg.solve(design, 10.0 * preference_signs)
    with torch.no_grad():
        actor.advantage.weight.fill_(weight)
        actor.advantage.bias.fill_(bias)
    assert actor.preference_loss(latents, rewards).item() < 0.001
    return actor


@pytest.mark.parametrize("invalid_reward", [torch.nan, torch.inf, -torch.inf])
def test_binary_actor_rejects_nonfinite_preference_targets(invalid_reward) -> None:
    actor = BinaryAdvantageHead(latent_dim=1)
    latents = torch.tensor([[0.0]])
    rewards = torch.tensor([[invalid_reward, 0.0]])

    with pytest.raises(ValueError, match="must be finite"):
        actor.preference_loss(latents, rewards)


@pytest.mark.parametrize("invalid_latent", [torch.nan, torch.inf, -torch.inf])
def test_binary_actor_forward_rejects_nonfinite_latents(invalid_latent) -> None:
    actor = BinaryAdvantageHead(latent_dim=1)
    latents = torch.tensor([[invalid_latent]])

    with pytest.raises(ValueError, match="latent_z must be finite"):
        actor.forward(latents)


@pytest.mark.parametrize("invalid_latent", [torch.nan, torch.inf, -torch.inf])
def test_binary_actor_loss_rejects_nonfinite_latents(invalid_latent) -> None:
    actor = BinaryAdvantageHead(latent_dim=1)
    latents = torch.tensor([[invalid_latent]])
    rewards = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="latent_z must be finite"):
        actor.preference_loss(latents, rewards)


@pytest.mark.parametrize("invalid_latent", [torch.nan, torch.inf, -torch.inf])
def test_binary_actor_action_selection_rejects_nonfinite_latents(
    invalid_latent,
) -> None:
    actor = BinaryAdvantageHead(latent_dim=1)
    latents = torch.tensor([[invalid_latent]])

    with pytest.raises(ValueError, match="latent_z must be finite"):
        actor.select_action(latents)


def test_perfect_next_observation_prediction_is_not_control_sufficient() -> None:
    """Identical prediction signatures cannot encode opposite action preferences."""
    observer = _exact_two_state_observer()
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    predictions = torch.empty((2, 2, 1))
    actor_rewards = torch.empty_like(rewards)
    latents: list[float] = []
    latent_rows: list[torch.Tensor] = []

    for state in (0, 1):
        observation = torch.tensor([float(state)])
        latent = observer.encode(observation)
        latents.append(float(latent.item()))
        latent_rows.append(latent.detach().reshape(1))

        for action in (0, 1):
            action_tensor = torch.tensor(action, dtype=torch.long)
            predictions[state, action] = observer.predict(latent, action_tensor)

            transition = Transition(
                obs=np.array([state], dtype=np.float32),
                action=action,
                reward=float(rewards[state, action]),
                next_obs=np.array([0], dtype=np.float32),
                done=False,
                latent_z=latent.detach().numpy(),
                next_latent_z=observer.encode(torch.tensor([0.0])).detach().numpy(),
            )
            observer_batch = transition.to_observer_batch()
            actor_batch = transition.to_actor_batch()

            assert observer_batch
            assert set(observer_batch) == {"obs", "action", "next_obs"}
            torch.testing.assert_close(
                observer_batch["obs"], observation, rtol=0, atol=0
            )
            torch.testing.assert_close(
                observer_batch["action"], action_tensor, rtol=0, atol=0
            )
            torch.testing.assert_close(
                observer_batch["next_obs"], torch.tensor([0.0]), rtol=0, atol=0
            )

            assert set(actor_batch) == {
                "latent_z",
                "action",
                "reward",
                "next_latent_z",
                "done",
            }
            assert {"obs", "next_obs"}.isdisjoint(actor_batch)
            torch.testing.assert_close(actor_batch["latent_z"], latent, rtol=0, atol=0)
            torch.testing.assert_close(
                actor_batch["action"], action_tensor, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actor_batch["reward"], rewards[state, action], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actor_batch["next_latent_z"],
                torch.tensor([0.0]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actor_batch["done"], torch.tensor(0.0), rtol=0, atol=0
            )
            actor_rewards[state, action] = actor_batch["reward"]

    true_next_observations = torch.zeros_like(predictions)
    prediction_signatures = predictions.squeeze(-1)

    assert latents == [0.0, 1.0]
    assert torch.equal(predictions, true_next_observations)
    assert torch.equal(prediction_signatures[0], prediction_signatures[1])
    assert torch.equal(actor_rewards, rewards)
    assert rewards[0, 0] - rewards[0, 1] == 1
    assert rewards[1, 1] - rewards[1, 0] == 1
    assert rewards.argmax(dim=1).tolist() == [0, 1]

    # Any deterministic prediction-signature-only selector must choose one
    # common action and is correct in at most one of the two states.
    optimal_actions = rewards.argmax(dim=1)
    prediction_only_accuracies = [
        float(
            (torch.full_like(optimal_actions, action) == optimal_actions).float().mean()
        )
        for action in (0, 1)
    ]
    assert max(prediction_only_accuracies) == 0.5

    latent_batch = torch.stack(latent_rows)
    actor = _fit_binary_actor(latent_batch, actor_rewards)
    assert actor.select_action(latent_batch).tolist() == [0, 1]

    # Identical dynamics and latents with the reward table reversed force the
    # learned policy to flip, ruling out an accidental action=latent shortcut.
    flipped_rewards = actor_rewards.flip(dims=(1,))
    flipped_actor = _fit_binary_actor(latent_batch, flipped_rewards)
    assert flipped_actor.select_action(latent_batch).tolist() == [1, 0]
