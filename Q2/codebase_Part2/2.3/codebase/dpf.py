import tensorflow as tf

class DifferentiableParticleFilter(tf.Module):
    def __init__(self, model, num_particles, resampling_module):
        super().__init__()
        self.model = model
        self.num_particles = num_particles
        self.resampler = resampling_module

    def __call__(self, observations):
        """
        Runs the PF over the sequence of observations.
        observations: [Batch, Time, Dim]
        """
        batch_size = tf.shape(observations)[0]
        time_steps = tf.shape(observations)[1]
        state_dim = 1 # Simple 1D model

        # Initialization
        particles = tf.random.normal([batch_size, self.num_particles, state_dim])
        log_weights = tf.fill([batch_size, self.num_particles], -tf.math.log(float(self.num_particles)))

        loss = 0.0

        for t in range(time_steps):
            current_obs = observations[:, t, :]

            # 1. Resampling (Corenflos paper usually puts resampling first or last depending on scheme)
            # Here we resample at the start of step t (except t=0 ideally, but for simplicity we run it)
            particles, log_weights = self.resampler(log_weights, particles)

            # 2. Transition (Prediction)
            particles = self.model.transition(particles)

            # 3. Weighting (Correction)
            # Log Likelihood p(y_t | x_t)
            log_lik = self.model.observation_log_prob(particles, current_obs)
            log_lik = tf.squeeze(log_lik, -1) # [B, N]
            
            log_weights = log_weights + log_lik

            # Normalize weights for stability
            lse = tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
            log_weights = log_weights - lse
            
            # Simple ELBO-like loss approximation (negative log likelihood estimate)
            # Loss = - log( sum(w * p(y)) ) approx -LSE
            step_loss = -tf.reduce_mean(lse)
            loss += step_loss

        return loss