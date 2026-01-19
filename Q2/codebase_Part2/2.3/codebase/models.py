import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class LinearGaussianSSM(tf.Module):
    """
    A simple 1D Linear Gaussian State Space Model.
    x_t = a * x_{t-1} + noise
    y_t = x_t + noise
    """
    def __init__(self, transition_a=0.5, obs_noise_std=1.0):
        super().__init__()
        # We make 'transition_a' a Variable so we can calculate gradients w.r.t it.
        self.transition_a = tf.Variable(transition_a, dtype=tf.float32, name="transition_param")
        self.obs_noise_std = tf.constant(obs_noise_std, dtype=tf.float32)

    def transition(self, particles):
        # particles: [B, N, 1]
        noise = tf.random.normal(tf.shape(particles))
        return self.transition_a * particles + noise

    def observation_log_prob(self, particles, observation):
        # particles: [B, N, 1]
        # observation: [B, 1]
        obs_broadcast = tf.expand_dims(observation, 1) # [B, 1, 1]
        dist = tfd.Normal(loc=particles, scale=self.obs_noise_std)
        return dist.log_prob(obs_broadcast) # [B, N, 1]