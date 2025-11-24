import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ==========================================
# 1.  Data Generation)
# ==========================================
#  A = L + E

FIRMS = ['Tencent', 'JPMorgan', 'Alibaba_HK', 'Exxon', 'Volkswagen', 'Microsoft', 'Google']
# 5 Assets, 5 Liabs, 4 Equity
n_assets = 5
n_liabs = 5
n_equity = 4
n_bottom_features = n_assets + n_liabs + n_equity
n_firms = len(FIRMS)
n_years = 5

def generate_balanced_data():
    np.random.seed(42)
    # 1. Generate Liabilities and Equity randomly
    L_E_data = np.random.uniform(10, 50, size=(n_firms, n_years, n_liabs + n_equity))
    
    # 2. Calculate Total L + E
    # Sum across the feature axis
    Total_LE = np.sum(L_E_data, axis=2)
    
    # 3. Generate Asset Components that SUM up to Total_LE
    # We generate random weights for assets and normalize them
    raw_asset_weights = np.random.uniform(0.5, 1.5, size=(n_firms, n_years, n_assets))
    weight_sums = np.sum(raw_asset_weights, axis=2, keepdims=True)
    normalized_weights = raw_asset_weights / weight_sums
    
    # Distribute Total_LE into Asset components
    A_data = normalized_weights * Total_LE[:, :, np.newaxis]
    
    # 4. Concatenate: [Assets, Liabs, Equity]
    # Structure: First 5 are Assets, Next 5 Liabs, Last 4 Equity
    data = np.concatenate([A_data, L_E_data], axis=2)
    return data.astype(np.float32)

raw_data_tensor = generate_balanced_data()

# Verify the synthetic data IS balanced
sample = raw_data_tensor[0, 0, :]
assert abs(np.sum(sample[:5]) - np.sum(sample[5:])) < 1e-4, "Data generation failed identity check!"

# ==========================================
# 2. Define Constraints and Matrix
# ==========================================
# Same Summing Matrix as before for Vertical Aggregation
def get_summing_matrix():
    S = np.zeros((3, n_bottom_features))
    # Asset Row (0-4)
    S[0, :n_assets] = 1
    # Liab Row (5-9)
    S[1, n_assets : n_assets + n_liabs] = 1
    # Equity Row (10-13)
    S[2, n_assets + n_liabs :] = 1
    # Full Hierarchy: [Aggregates; Components]
    S_full = np.vstack([S, np.eye(n_bottom_features)])
    return tf.constant(S_full, dtype=tf.float32)

S_MATRIX_FIRM = get_summing_matrix()

# ==========================================
# 3. The Hard Constraint Layer
# ==========================================

class BalanceSheetConstraintLayer(layers.Layer):
    """
    This layer forces the 'Horizontal' identity:
    Sum(Assets) - Sum(Liabs) - Sum(Equity) = 0
    It effectively projects the prediction onto the valid subspace.
    """
    def __init__(self, **kwargs):
        super(BalanceSheetConstraintLayer, self).__init__(**kwargs)
        # We create a mask to easily sum components
        # +1 for Assets, -1 for Liabs, -1 for Equity
        self.constraint_vector = np.zeros((n_bottom_features, 1), dtype=np.float32)
        self.constraint_vector[:n_assets] = 1.0
        self.constraint_vector[n_assets:] = -1.0 
        self.constraint_vector = tf.constant(self.constraint_vector)

    def call(self, inputs):
        # inputs: Predicted Deltas or Values (Batch, Features)
        
        # 1. Calculate the Discrepancy (Gap)
        # Gap = Sum(A) - Sum(L) - Sum(E)
        gap = tf.matmul(inputs, self.constraint_vector) # Shape (Batch, 1)
        
        # 2. Distribute the Gap to eliminate it.
        # Simple Logic: Subtract Gap/N from Assets, Add Gap/N to Liabs/Equity
        # N = Total number of components (14)
        # This spreads the error evenly across all fields to satisfy the equation.
        
        n_total = float(n_bottom_features)
        adjustment = gap / n_total
        
        # We need to subtract adjustment from A (where vec is +1)
        # and add adjustment to L/E (where vec is -1) to close the gap.
        # Actually, simpler math: 
        # New_Val = Old_Val - (Constraint_Vector * Adjustment)
        # Check: If Gap is positive (Assets too high), Vector is +1 for Assets -> We subtract. 
        # Vector is -1 for Liabs -> We add ( - (-Adj) = +Adj).
        
        adjustment_vector = tf.matmul(adjustment, tf.transpose(self.constraint_vector))
        
        corrected_inputs = inputs - adjustment_vector
        return corrected_inputs

# ==========================================
# 4. Building the Improved Model
# ==========================================

class VerticalReconciliationLayer(layers.Layer):
    def __init__(self, s_matrix, **kwargs):
        super(VerticalReconciliationLayer, self).__init__(**kwargs)
        self.s_matrix = s_matrix

    def call(self, inputs):
        return tf.matmul(inputs, self.s_matrix, transpose_b=True)

def build_constrained_model(n_features):
    inputs = layers.Input(shape=(n_features,))
    
    # 1. Predict raw changes (Deltas)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    raw_delta = layers.Dense(n_features, name='raw_delta')(x)
    
    # 2. Apply HORIZONTAL Constraint (Force A - L - E = 0 on the deltas)
    # If the change in A equals change in L+E, the balance is preserved.
    balanced_delta = BalanceSheetConstraintLayer(name='balanced_delta')(raw_delta)
    
    # 3. Add to Input (Residual Connection)
    bottom_prediction = layers.Add()([inputs, balanced_delta])
    
    # 4. Apply VERTICAL Reconciliation (Get Totals)
    coherent_output = VerticalReconciliationLayer(S_MATRIX_FIRM, name='coherent_out')(bottom_prediction)
    
    model = Model(inputs=inputs, outputs=coherent_output)
    return model

# ==========================================
# 5. Train & Test
# ==========================================

# Prepare Data (Using the new Balanced Data)
def prepare_paired_data(data):
    X_list, Y_list = [], []
    for f in range(data.shape[0]):
        firm_data = data[f]
        for t in range(data.shape[1] - 1):
            X_list.append(firm_data[t])
            Y_list.append(firm_data[t+1])
    return np.array(X_list), np.array(Y_list)

train_data_slice = raw_data_tensor[:, :-1, :]
X_train, Y_train = prepare_paired_data(train_data_slice)

# Prepare Target (Projected through S matrix)
Y_train_coherent = tf.matmul(Y_train, S_MATRIX_FIRM, transpose_b=True)

model = build_constrained_model(n_bottom_features)
model.compile(optimizer='adam', loss='mse')

print("Training Improved Model...")
model.fit(X_train, Y_train_coherent, epochs=100, batch_size=4, verbose=0)

print("\n--- Testing on Year 5 ---")
# Prepare Test
X_test = raw_data_tensor[:, -2, :] # Year 4
Y_test = raw_data_tensor[:, -1, :] # Year 5

preds = model.predict(X_test)

# Verify Identity
pred_A = preds[:, 0]
pred_L = preds[:, 1]
pred_E = preds[:, 2]

identity_errors = np.abs(pred_A - (pred_L + pred_E))
print(f"Max Identity Error in Test Set: {np.max(identity_errors):.10f}") # Should be near 0

global_gap = np.sum(pred_A) - (np.sum(pred_L) + np.sum(pred_E))
print(f"Global Level 0 Identity Gap: {global_gap:.10f}") # Should be near 0