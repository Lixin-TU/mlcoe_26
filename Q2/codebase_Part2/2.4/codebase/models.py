import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class LinearGaussianSSM(tf.Module):
    """
    A simple 1D Linear Gaussian State Space Model (LGSSM).
    Used as the testbed for gradient analysis.
    """
    def __init__(self, transition_a=0.5, obs_noise_std=1.0):
        super().__init__()
        # We make 'transition_a' a Variable to compute gradients w.r.t it.
        self.transition_a = tf.Variable(transition_a, dtype=tf.float32, name="transition_param")
        self.obs_noise_std = tf.constant(obs_noise_std, dtype=tf.float32)

    def transition(self, particles):
        """
        Propagate particles: x_t = a * x_{t-1} + noise
        Args:
            particles: [Batch, N, 1]
        """
        noise = tf.random.normal(tf.shape(particles))
        return self.transition_a * particles + noise

    def observation_log_prob(self, particles, observation):
        """
        Compute log p(y_t | x_t).
        Args:
            particles: [Batch, N, 1]
            observation: [Batch, 1]
        """
        # Broadcast observation to compare with all particles
        obs_broadcast = tf.expand_dims(observation, 1) # [B, 1, 1]
        dist = tfd.Normal(loc=particles, scale=self.obs_noise_std)
        return dist.log_prob(obs_broadcast) # [B, N, 1]