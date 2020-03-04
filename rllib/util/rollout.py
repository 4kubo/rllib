"""Helper functions to conduct a rollout with policies or agents."""

import torch
from rllib.dataset.datatypes import Observation
import pickle

__all__ = ['rollout_agent', 'rollout_policy', 'rollout_model']


def _step(environment, state, action, render):
    try:
        next_state, reward, done, _ = environment.step(action)
    except TypeError:
        next_state, reward, done, _ = environment.step(action.item())
    observation = Observation(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              done=done)
    state = next_state
    if render:
        environment.render()
    return observation, state, done


def rollout_agent(environment, agent, num_episodes=1, max_steps=1000, render=False,
                  milestones=None):
    """Conduct a single rollout of an agent in an environment.

    Parameters
    ----------
    environment: AbstractEnvironment
    agent: AbstractAgent
    num_episodes: int, optional (default=1)
    max_steps: int.
    render: bool.
    milestones: list.
        List with episodes in which to save the agent.

    """
    milestones = list() if milestones is None else milestones
    for episode in range(num_episodes):
        state = environment.reset()
        agent.start_episode()
        done = False
        while not done:
            action = agent.act(state)
            observation, state, done = _step(environment, state, action, render)
            agent.observe(observation)
            if max_steps <= environment.time:
                break
        agent.end_episode()

        if episode in milestones:
            file_name = '{}_{}_{}.pkl'.format(environment.name, agent.name, episode)
            with open(file_name, 'wb') as file:
                pickle.dump(agent, file)
    agent.end_interaction()


def rollout_policy(environment, policy, num_episodes=1, max_steps=1000, render=False):
    """Conduct a single rollout of a policy in an environment.

    Parameters
    ----------
    environment : AbstractEnvironment
    policy : AbstractPolicy
    num_episodes: int, optional (default=1)
    max_steps: int
    render: bool

    Returns
    -------
    trajectories: list of list of Observation.

    """
    trajectories = []
    for _ in range(num_episodes):
        state = environment.reset()
        done = False
        trajectory = []
        with torch.no_grad():
            while not done:
                action = policy(torch.tensor(state).float()).sample().numpy()
                observation, state, done = _step(environment, state, action, render)
                trajectory.append(observation)
                if max_steps <= environment.time:
                    break
        trajectories.append(trajectory)
    return trajectories


def rollout_model(model, policy, initial_states, max_steps=1000, termination=None):
    """Rollout a system for a number of steps."""
    trajectory = list()
    states = initial_states
    for _ in range(max_steps):
        # Sample actions
        actions = policy(states)
        if actions.has_rsample:
            actions = actions.rsample()
        else:
            actions = actions.sample()

        # Sample next states
        next_states = model(states, actions)
        if next_states.has_rsample:
            next_states = next_states.rsample()
        else:
            next_states = next_states.sample()

        # Store state, action tuples
        trajectory.append(Observation(states, actions))

        # Update state
        states = next_states

        # Check for termination.
        if termination is not None and termination(states, actions):
            break

    return trajectory
