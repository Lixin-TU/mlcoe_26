import tensorflow as tf

class SoftResampling(tf.Module):
    """
    Mixes particle weights with a uniform distribution before resampling.
    """
    def __init__(self, alpha: float, name=None):
        super(SoftResampling, self).__init__(name=name)
        # alpha = 1.0 -> Hard Resampling (Standard)
        # alpha = 0.0 -> Uniform Sampling (No information kept)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def __call__(self, log_weights, particles):
        """
        Args:
            log_weights: [batch_size, num_particles]
            particles: [batch_size, num_particles, state_dim]
        """
        num_particles = tf.shape(log_weights)[1]
        batch_size = tf.shape(log_weights)[0]

        # 1. Normalize weights (Gradient flows here)
        weights = tf.nn.softmax(log_weights, axis=-1)

        # 2. Mix with Uniform Distribution
        # w_soft = alpha * w + (1-alpha) * (1/N)
        uniform_weights = tf.ones_like(weights) / tf.cast(num_particles, tf.float32)
        soft_weights = self.alpha * weights + (1.0 - self.alpha) * uniform_weights

        # 3. Sample Indices (Multinomial)
        logits = tf.math.log(soft_weights + 1e-9) 
        indices = tf.random.categorical(logits, num_samples=num_particles, dtype=tf.int32)

        # 4. Gather particles
        # Create batch indices for gather_nd
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_particles])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        
        new_particles = tf.gather_nd(particles, gather_indices)

        # 5. Reset weights to 1/N
        new_log_weights = tf.fill(tf.shape(log_weights), -tf.math.log(tf.cast(num_particles, tf.float32)))

        return new_particles, new_log_weights