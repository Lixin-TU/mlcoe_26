import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import os

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 0. Load Shared Data
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'balance_sheet_data_Q2_2.csv')

if not os.path.exists(csv_path):
    print(f"Error: Data file {csv_path} not found. Please run run_forecast.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# Meta info
FIRMS = df['Firm'].unique()
feature_cols = ['Cash_Equiv', 'Acct_Rec', 'Net_PPE', 'Avail_Sale_Sec', 'Other_Assets',
                'Curr_Debt', 'Long_Term_Debt', 'Payables', 'Def_Tax_Liab', 'Other_Liab',
                'Capital_Stock', 'Common_Stock', 'Ret_Earnings', 'Other_Equity']

n_features = len(feature_cols)
n_assets = 5 # First 5 are assets

# Preprocess into Tensor format: (n_firms, n_years, n_features)
# Sort by Firm and Year to ensure correct order
df_sorted = df.sort_values(['Firm', 'Year'])
n_firms = len(FIRMS)
n_years = df['Year'].nunique()

# Verify structure
if len(df) != n_firms * n_years:
    print("Warning: Data might be missing rows.")

data_tensor = np.zeros((n_firms, n_years, n_features), dtype=np.float32)

for i, firm in enumerate(FIRMS):
    firm_df = df_sorted[df_sorted['Firm'] == firm]
    data_tensor[i, :, :] = firm_df[feature_cols].values

print(f"Data Loaded. Shape: {data_tensor.shape} (Firms, Years, Features)")

# ==========================================
# 1. Prepare Train/Test Split
# ==========================================
# Task: Predict Year(t+1) given Year(t)
# Train: Years 0->1, 1->2, 2->3
# Test: Year 3->4

# Flatten for models
X_train_list = []
Y_train_target_list = [] # Absolute values for Linear Regression
Y_train_delta_list = []  # Deltas for MLP

X_test_list = []
Y_test_target_list = []
Y_test_delta_list = []

for f in range(n_firms):
    # Training Pairs
    for t in range(0, 3): 
        state_t = data_tensor[f, t, :]
        state_next = data_tensor[f, t+1, :]
        
        X_train_list.append(state_t)
        Y_train_target_list.append(state_next)
        Y_train_delta_list.append(state_next - state_t)

    # Test Pair (Year 3 to 4)
    state_t = data_tensor[f, 3, :]
    state_next = data_tensor[f, 4, :]
    
    X_test_list.append(state_t)
    Y_test_target_list.append(state_next)
    Y_test_delta_list.append(state_next - state_t)

X_train = np.array(X_train_list)
Y_train_target = np.array(Y_train_target_list)
Y_train_delta = np.array(Y_train_delta_list)

X_test = np.array(X_test_list)
Y_test_target = np.array(Y_test_target_list)
Y_test_delta = np.array(Y_test_delta_list)

print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ==========================================
# 2. LLM Model (Linear Regression)
# ==========================================
# The agent used manual Normal Equation via Numpy
print("\n--- Training LLM Selected Model (Linear Regression) ---")

# Add bias
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Theta = (X^T X + lambda*I)^-1 X^T Y
# Using small regularization as agent did
identity_matrix = np.eye(X_train_bias.shape[1])
try:
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias + 1e-6 * identity_matrix) @ X_train_bias.T @ Y_train_target
except np.linalg.LinAlgError:
    theta = np.linalg.pinv(X_train_bias) @ Y_train_target

Y_pred_LLM = X_test_bias @ theta

mse_llm = np.mean((Y_test_target - Y_pred_LLM) ** 2)
print(f"LLM Model (Linear Regression) MSE: {mse_llm:.4f}")


# ==========================================
# 3. Part 1 Model (MLP with Constraint)
# ==========================================
print("\n--- Training Part 1 Model (MLP + Constraint) ---")

# Recreate the logic from Q1_6.py
constraint_vector = np.zeros((n_features, 1), dtype=np.float32)
constraint_vector[:n_assets] = 1.0
constraint_vector[n_assets:] = -1.0 
constraint_vector_tf = tf.constant(constraint_vector)

class BalanceSheetConstraintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BalanceSheetConstraintLayer, self).__init__(**kwargs)
        self.constraint_vector = constraint_vector_tf

    def call(self, inputs):
        # inputs: Predicted Deltas
        gap = tf.matmul(inputs, self.constraint_vector)
        adjustment = gap / float(n_features)
        adjustment_vector = tf.matmul(adjustment, tf.transpose(self.constraint_vector))
        return inputs - adjustment_vector

def build_part1_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    raw_delta = layers.Dense(input_dim, name='raw_delta')(x)
    balanced_delta = BalanceSheetConstraintLayer(name='balanced_delta')(raw_delta)
    return Model(inputs=inputs, outputs=balanced_delta)

model_p1 = build_part1_model(n_features)
model_p1.compile(optimizer='adam', loss='mse')

# Train on Deltas
# Using more epochs since small dataset
model_p1.fit(X_train, Y_train_delta, epochs=200, batch_size=4, verbose=0)

# Predict Deltas
pred_deltas_p1 = model_p1.predict(X_test, verbose=0)

# Convert to Absolute Values: Pred_State = Current_State + Pred_Delta
Y_pred_P1 = X_test + pred_deltas_p1

mse_p1 = np.mean((Y_test_target - Y_pred_P1) ** 2)
print(f"Part 1 Model (MLP) MSE: {mse_p1:.4f}")

# ==========================================
# 4. Ensemble Model
# ==========================================
print("\n--- Training Ensemble Model (Simple Average) ---")

# Average predictions
Y_pred_Ensemble = (Y_pred_LLM + Y_pred_P1) / 2.0

mse_ensemble = np.mean((Y_test_target - Y_pred_Ensemble) ** 2)
print(f"Ensemble Model MSE: {mse_ensemble:.4f}")

# ==========================================
# 5. Conclusion
# ==========================================
print("\n=== SUMMARY RESULTS ===")
print(f"LLM Model MSE:      {mse_llm:.4f}")
print(f"Part 1 Model MSE:   {mse_p1:.4f}")
print(f"Ensemble Model MSE: {mse_ensemble:.4f}")

better_model = "LLM" if mse_llm < mse_p1 else "Part 1 Model"
print(f"\nQuestion b) The {better_model} performs better.")

ensemble_improvement = min(mse_llm, mse_p1) - mse_ensemble
is_ensemble_better = ensemble_improvement > 0
print(f"Question c) Ensemble is better? {is_ensemble_better} (Improvement: {ensemble_improvement:.4f})")
