import tensorflow as tf

class DifferentiableParticleFilter(tf.Module):
    def __init__(self, model, num_particles, resampling_module):
        super().__init__()
        self.model = model
        self.num_particles = num_particles
        self.resampler = resampling_module

    def __call__(self, observations):
        """
        Run DPF on observations [B, T, D]
        """
        batch_size = tf.shape(observations)[0]
        time_steps = tf.shape(observations)[1]
        
        # Init
        particles = tf.random.normal([batch_size, self.num_particles, 1])
        log_weights = tf.fill([batch_size, self.num_particles], -tf.math.log(float(self.num_particles)))
        loss = 0.0

        for t in range(time_steps):
            # 1. Resampling (Soft or OT)
            particles, log_weights = self.resampler(log_weights, particles)
            
            # 2. Transition
            particles = self.model.transition(particles)
            
            # 3. Weighting
            current_obs = observations[:, t, :]
            log_lik = self.model.observation_log_prob(particles, current_obs)
            log_lik = tf.squeeze(log_lik, -1)
            log_weights = log_weights + log_lik
            
            # Normalize & Accumulate pseudo-loss (negative log evidence)
            lse = tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
            log_weights = log_weights - lse
            loss += -tf.reduce_mean(lse)

        return loss