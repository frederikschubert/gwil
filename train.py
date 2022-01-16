import coax
import gym
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
from coax.value_losses import mse
from omegaconf import DictConfig, OmegaConf
from optax import adam
from ott.core import gromov_wasserstein as gw
from ott.geometry import pointcloud


def expert_policy(state: jnp.ndarray):
    w = state[3]
    return 0 if w < 0 else 1


def expert_policy_v2(state: jnp.ndarray):
    theta = state[2]
    return 0 if theta < 0 else 1


@hydra.main("./configs", "base")
def train(cfg: DictConfig):
    env = gym.make(cfg.env)
    env = coax.wrappers.TrainMonitor(env)

    print(OmegaConf.to_yaml(cfg))
    print(env)
    print(f"Observation Space: ", env.observation_space)
    print(f"Action Space: ", env.action_space)

    expert_trajectory = []

    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = expert_policy(s)
        s_next, r, done, _ = env.step(a)
        expert_trajectory.append(jnp.concatenate([s, jnp.array([a])]))
        s = s_next
        if done:
            break
    # coax.utils.generate_gif(env=env, policy=expert_policy, filepath=f"expert.gif")
    env.close()
    expert_trajectory = jnp.stack(expert_trajectory)
    print(expert_trajectory.shape)

    env = gym.make(cfg.env)
    env = coax.wrappers.TrainMonitor(env, tensorboard_dir="./")

    def func(S, is_training):
        """type-2 q-function: s -> q(s,.)"""
        seq = hk.Sequential(
            (
                hk.Linear(128),
                jax.nn.relu,
                hk.Linear(128),
                jax.nn.relu,
                hk.Linear(env.action_space.n, w_init=jnp.zeros),
            )
        )
        return seq(S)

    # value function and its derived policy
    q = coax.Q(func, env)
    pi = coax.BoltzmannPolicy(q, temperature=0.1)

    # target network
    q_targ = q.copy()

    # experience tracer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
    buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

    # updater
    qlearning = coax.td_learning.QLearning(
        q, q_targ=q_targ, loss_function=mse, optimizer=adam(0.001)
    )

    # train
    for ep in range(1000):
        trajectory = []
        s = env.reset()
        # pi.epsilon = max(0.01, pi.epsilon * 0.95)
        # env.record_metrics({'EpsilonGreedy/epsilon': pi.epsilon})

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)
            trajectory.append((s, a, r, done))

            # learn
            if len(buffer) >= 100:
                transition_batch = buffer.sample(batch_size=32)
                metrics = qlearning.update(transition_batch)
                env.record_metrics(metrics)

            # sync target network
            q_targ.soft_update(q, tau=0.01)

            if done:
                # compute coupling
                agent_trajectory = jnp.stack(
                    [jnp.concatenate([s, jnp.array([a])]) for s, a, *_ in trajectory]
                )
                geom_x = pointcloud.PointCloud(x=agent_trajectory)
                geom_y = pointcloud.PointCloud(x=expert_trajectory)
                out = gw.gromov_wasserstein(
                    geom_x=geom_x,
                    geom_y=geom_y,
                    epsilon=1.0,
                    jit=True,
                )
                # compute rewards
                proxy_reward = -jnp.sum(out.transport * out.cost_matrix, axis=-1)
                env.record_metrics(
                    {
                        "gwil/gw_cost": out.reg_gw_cost,
                        "gwil/proxy_reward": jnp.mean(proxy_reward),
                    }
                )

                # add transitions
                for transition, il_r in zip(trajectory, proxy_reward):
                    s, a, r, done = transition
                    tracer.add(s, a, il_r, done)
                    while tracer:
                        buffer.add(tracer.pop())
                break

            s = s_next


if __name__ == "__main__":
    train()
