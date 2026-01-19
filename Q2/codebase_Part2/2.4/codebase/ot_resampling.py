import tensorflow as tf

class RegularisedTransform(tf.Module):
    """
    Implements Entropy-Regularized Optimal Transport Resampling (Sinkhorn Algorithm).
    Uses the Standard Scaling formulation for numerical stability.
    """
    def __init__(self, epsilon, max_iter=100, threshold=1e-2, name=None):
        super(RegularisedTransform, self).__init__(name=name)
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.max_iter = tf.constant(max_iter, dtype=tf.int32)
        self.threshold = tf.constant(threshold, dtype=tf.float32)

    def _cost_matrix(self, x, y):
        # Squared Euclidean Distance: ||x - y||^2
        x_col = tf.expand_dims(x, 2)
        y_row = tf.expand_dims(y, 1)
        return tf.reduce_sum(tf.square(x_col - y_row), axis=-1)

    def _sinkhorn_potentials(self, log_alpha, log_beta, cost_mat):
        """
        Computes potentials f and g such that P = exp((f + g - C) / epsilon)
        satisfies the marginal constraints.
        """
        batch_size = tf.shape(cost_mat)[0]
        n = tf.shape(cost_mat)[1]
        
        # Initialize potentials
        f = tf.zeros([batch_size, n], dtype=tf.float32)
        g = tf.zeros([batch_size, n], dtype=tf.float32)
        
        def body(i, f, g, err):
            # Standard Sinkhorn Update in Log-Domain
            # f <- eps * log(alpha) - eps * LSE( (g - C) / eps )
            # g <- eps * log(beta)  - eps * LSE( (f - C) / eps )
            
            # 1. Update f (rows)
            # Kernel: (g - C) / eps
            tmp = (tf.expand_dims(g, 1) - cost_mat) / self.epsilon
            lse_f = tf.reduce_logsumexp(tmp, axis=2)
            # Damped update for stability: f = 0.5*f + 0.5*new_f
            f_target = self.epsilon * log_alpha - self.epsilon * lse_f
            f_new = 0.5 * (f + f_target)
            
            # 2. Update g (cols)
            # Kernel: (f - C) / eps
            tmp = (tf.expand_dims(f_new, 2) - cost_mat) / self.epsilon
            lse_g = tf.reduce_logsumexp(tmp, axis=1)
            g_target = self.epsilon * log_beta - self.epsilon * lse_g
            g_new = 0.5 * (g + g_target)
            
            # Compute convergence error
            err = tf.reduce_max(tf.abs(f_new - f)) + tf.reduce_max(tf.abs(g_new - g))
            
            return i + 1, f_new, g_new, err

        def cond(i, f, g, err):
            return tf.logical_and(i < self.max_iter, err > self.threshold)

        _, f, g, _ = tf.while_loop(cond, body, [0, f, g, 10.0])
        return f, g

    def __call__(self, log_weights, particles):
        n = tf.shape(log_weights)[1]
        batch_size = tf.shape(log_weights)[0]
        
        # Target (beta): Particle weights; Source (alpha): Uniform
        log_beta = tf.nn.log_softmax(log_weights, axis=-1)
        log_alpha = tf.fill([batch_size, n], -tf.math.log(tf.cast(n, tf.float32)))
        
        cost_mat = self._cost_matrix(particles, particles)
        
        # Run Sinkhorn
        f, g = self._sinkhorn_potentials(log_alpha, log_beta, cost_mat)
        
        # Compute Transport Matrix P
        # P_ij = exp( (f_i + g_j - C_ij) / epsilon )
        log_P = (tf.expand_dims(f, 2) + tf.expand_dims(g, 1) - cost_mat) / self.epsilon
        P = tf.exp(log_P)
        
        # Barycentric Projection: X_new = N * P * X_old
        # P maps indices (New) -> locations (Old)
        # Check normalization implicitly: Sum_j P_ij should be alpha_i = 1/N.
        # So (N * P) sums to 1.
        new_particles = tf.cast(n, tf.float32) * tf.matmul(P, particles)
        
        return new_particles, log_alpha