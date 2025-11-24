import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# 1. Generate Synthetic Balance Sheet Data
# ---------------------------------------------------------
def generate_data(n_samples=1000):
    # Random Liabilities and Equity
    L = np.random.uniform(50, 100, n_samples)
    E = np.random.uniform(10, 50, n_samples)
    
    # Assets must equal L + E (Accounting Identity)
    A = L + E
    
    # Stack into a feature matrix [A, L, E]
    data = np.stack([A, L, E], axis=1)
    
    # Add some noise to simulating "inputs" vs "targets"
    X = data + np.random.normal(0, 2, data.shape) # Noisy past data
    Y = data # Clean ground truth
    return X, Y

X_train, Y_train = generate_data()

# 2. Define the Reconciliation Layer (Method B)
# ---------------------------------------------------------
class BalanceSheetReconciliation(layers.Layer):
    def __init__(self, **kwargs):
        super(BalanceSheetReconciliation, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch_size, 3) -> [A_pred, L_pred, E_pred]
        
        # Extract predicted components
        # A is index 0, L is index 1, E is index 2
        A_pred = inputs[:, 0:1]
        L_pred = inputs[:, 1:2]
        E_pred = inputs[:, 2:3]
        
        # Calculate the violation of the identity (the residual)
        # Identity: A - L - E = 0
        discrepancy = A_pred - (L_pred + E_pred)
        
        # Distribute the error equally (OLS Reconciliation)
        # We subtract from A and add to L and E to balance the equation
        A_rec = A_pred - (discrepancy / 3.0)
        L_rec = L_pred + (discrepancy / 3.0)
        E_rec = E_pred + (discrepancy / 3.0)
        
        # Concatenate back into a single tensor
        return tf.concat([A_rec, L_rec, E_rec], axis=1)

# 3. Build the Model
# ---------------------------------------------------------
def build_model():
    input_layer = layers.Input(shape=(3,))
    
    # --- Base Forecaster (Simple Dense Network) ---
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(32, activation='relu')(x)
    
    # Initial raw predictions (Unconstrained)
    base_predictions = layers.Dense(3, name='base_predictions')(x)
    
    #Apply Reconciliation
    reconciled_predictions = BalanceSheetReconciliation(name='reconciled_output')(base_predictions)
    
    model = Model(inputs=input_layer, outputs=reconciled_predictions)
    return model

# 4. Train and Test
# ---------------------------------------------------------
model = build_model()

model.compile(optimizer='adam', loss='mse')

print("Training Model...")
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
print("Training Complete.\n")

# 5. Verification Step
# ---------------------------------------------------------
# Let's run a prediction and check if A = L + E holds
sample_input = np.array([[150.0, 100.0, 50.0]]) # Rough estimates
prediction = model.predict(sample_input)

pred_A = prediction[0][0]
pred_L = prediction[0][1]
pred_E = prediction[0][2]

print(f"Predicted Assets (A):      {pred_A:.4f}")
print(f"Predicted Liabilities (L): {pred_L:.4f}")
print(f"Predicted Equity (E):      {pred_E:.4f}")

# Check the Identity
lhs = pred_A
rhs = pred_L + pred_E
gap = lhs - rhs

print(f"-"*30)
print(f"Identity Check (A - L - E): {gap:.10f}") 
print(f"Does Identity Hold?         {'YES' if abs(gap) < 1e-5 else 'NO'}")