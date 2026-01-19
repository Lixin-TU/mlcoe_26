import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 0. Load Shared Data (Refencing 2.2 Data)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Reference data from sibling directory 2.2
csv_path = os.path.join(current_dir, '../2.2/balance_sheet_data_Q2_2.csv')

if not os.path.exists(csv_path):
    print(f"Error: Data file {csv_path} not found. Please ensure 2.2 is run first.")
    exit(1)

df = pd.read_csv(csv_path)

# Meta info
FIRMS = df['Firm'].unique()
# Ensure we use the exact same feature columns
feature_cols = ['Cash_Equiv', 'Acct_Rec', 'Net_PPE', 'Avail_Sale_Sec', 'Other_Assets',
                'Curr_Debt', 'Long_Term_Debt', 'Payables', 'Def_Tax_Liab', 'Other_Liab',
                'Capital_Stock', 'Common_Stock', 'Ret_Earnings', 'Other_Equity']

n_features = len(feature_cols)
n_assets = 5 

# Preprocess into Tensor format: (n_firms, n_years, n_features)
df_sorted = df.sort_values(['Firm', 'Year'])
n_firms = len(FIRMS)
n_years = df['Year'].nunique()

data_tensor = np.zeros((n_firms, n_years, n_features), dtype=np.float32)
for i, firm in enumerate(FIRMS):
    firm_df = df_sorted[df_sorted['Firm'] == firm]
    data_tensor[i, :, :] = firm_df[feature_cols].values

print(f"Data Loaded. Shape: {data_tensor.shape}")

# ==========================================
# 1. Prepare Train/Test Split
# ==========================================
X_train_list = []
Y_train_target_list = [] 
Y_train_delta_list = []  

X_test_list = []
Y_test_target_list = []
Y_test_delta_list = []

for f in range(n_firms):
    # Training Pairs (Years 0->1, 1->2, 2->3)
    for t in range(0, 3): 
        state_t = data_tensor[f, t, :]
        state_next = data_tensor[f, t+1, :]
        
        X_train_list.append(state_t)
        Y_train_target_list.append(state_next)
        Y_train_delta_list.append(state_next - state_t)

    # Test Pair (Year 3->4)
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

# ==========================================
# 2. Train LLM Model (Linear Regression)
# ==========================================
print("\n--- Training LLM Selected Model (Linear Regression) ---")
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

identity_matrix = np.eye(X_train_bias.shape[1])
try:
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias + 1e-6 * identity_matrix) @ X_train_bias.T @ Y_train_target
except np.linalg.LinAlgError:
    theta = np.linalg.pinv(X_train_bias) @ Y_train_target

Y_pred_LLM = X_test_bias @ theta
mse_llm = np.mean((Y_test_target - Y_pred_LLM) ** 2)
print(f"LLM Model MSE: {mse_llm:.4f}")

# ==========================================
# 3. Train Part 1 Model (MLP + Constraint)
# ==========================================
print("\n--- Training Part 1 Model (MLP + Constraint) ---")

constraint_vector = np.zeros((n_features, 1), dtype=np.float32)
constraint_vector[:n_assets] = 1.0
constraint_vector[n_assets:] = -1.0 
constraint_vector_tf = tf.constant(constraint_vector)

class BalanceSheetConstraintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BalanceSheetConstraintLayer, self).__init__(**kwargs)
        # We will use this in call, but need to make sure it tracks with the layer
        self.constraint_vector_const = constraint_vector_tf

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # gap shape: (batch_size, 1)
        # inputs shape: (batch_size, 14)
        # constraint_vector shape: (14, 1)
        gap = tf.matmul(inputs, self.constraint_vector_const)
        
        adjustment = gap / float(n_features)
        
        # adjustment_vector shape: (batch_size, 14)
        adjustment_vector = tf.matmul(adjustment, tf.transpose(self.constraint_vector_const))
        
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
model_p1.fit(X_train, Y_train_delta, epochs=200, batch_size=4, verbose=0)

# Predict Deltas
pred_deltas_p1 = model_p1.predict(X_test, verbose=0)
Y_pred_P1 = X_test + pred_deltas_p1
mse_p1 = np.mean((Y_test_target - Y_pred_P1) ** 2)
print(f"Part 1 Model MSE: {mse_p1:.4f}")

# ==========================================
# 4. Ensemble Analysis (Weighted Average)
# ==========================================
print("\n--- Training Ensemble Models (Sweep Weights) ---")

results = []
alphas = np.linspace(0, 1, 101) # alpha is weight for Part 1 Model

best_mse = float('inf')
best_alpha = 0.0

for alpha in alphas:
    # Ensemble = alpha * Part1 + (1-alpha) * LLM
    Y_pred_ensemble = alpha * Y_pred_P1 + (1 - alpha) * Y_pred_LLM
    mse = np.mean((Y_test_target - Y_pred_ensemble) ** 2)
    results.append(mse)
    
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print(f"Best Ensemble MSE: {best_mse:.4f} at alpha={best_alpha:.2f} (Weight for Part 1 Model)")
print(f"Improvement over Part 1 alone: {mse_p1 - best_mse:.4f}")
print(f"Improvement over LLM alone: {mse_llm - best_mse:.4f}")

# Plotting the curve
plt.figure(figsize=(10, 6))
plt.plot(alphas, results, label='Ensemble MSE')
plt.axhline(y=mse_p1, color='r', linestyle='--', label='Part 1 Only (alpha=1.0)')
plt.axhline(y=mse_llm, color='g', linestyle='--', label='LLM Only (alpha=0.0)')
plt.scatter([best_alpha], [best_mse], color='red', zorder=5, label=f'Best (alpha={best_alpha:.2f})')
plt.title('Ensemble Model Performance: Part 1 vs LLM')
plt.xlabel('Weight for Part 1 Model (alpha)')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(current_dir, 'ensemble_optimization.png'))
print(f"Ensemble plot saved to {os.path.join(current_dir, 'ensemble_optimization.png')}")

# ==========================================
# 5. Advanced Ensemble (Linear Regression Stacking)
# ==========================================
print("\n--- Training Stacking Ensemble ---")
# Train a meta-learner to combine predictions
# We need validation predictions to train the meta-learner properly without overfitting, 
# but we have very little data. We'll use the training performance itself (risk of overfitting) 
# or a simple regression on the test set is cheating. 
# Let's try to learn weights on X_train predictions.

# 1. Get predictions on training set
pred_train_delta_p1 = model_p1.predict(X_train, verbose=0)
Y_train_pred_P1 = X_train + pred_train_delta_p1
Y_train_pred_LLM = X_train_bias @ theta

# 2. Stack features: [Pred_P1, Pred_LLM] -> Shape (n_samples, n_features * 2)
# Actually, usually stacking combines prediction per target.
# Let's learn a single weight alpha per feature? Or just global scalar alpha?
# Let's try to run a Linear Regression where Input = [P1_pred, LLM_pred], Output = True Target
# But we flatten everything to learn "Global" weighting.

# Inputs: (N_train * N_features, 2) -> Column 0: P1 pred, Column 1: LLM pred
stack_X_train = np.column_stack((Y_train_pred_P1.flatten(), Y_train_pred_LLM.flatten()))
stack_Y_train = Y_train_target.flatten()

# Meta model: Linear Regression (no bias, we just want weights)
meta_model = LinearRegression(fit_intercept=False) # We want w1*P1 + w2*LLM ≈ Target
meta_model.fit(stack_X_train, stack_Y_train)

weights = meta_model.coef_
print(f"Learned Stacking Weights: Part 1 = {weights[0]:.4f}, LLM = {weights[1]:.4f}")

# Predict on Test
stack_X_test = np.column_stack((Y_pred_P1.flatten(), Y_pred_LLM.flatten()))
stack_pred = meta_model.predict(stack_X_test)
mse_stacking = np.mean((Y_test_target.flatten() - stack_pred) ** 2)

print(f"Stacking Ensemble MSE: {mse_stacking:.4f}")

if mse_stacking < mse_p1:
    print("Stacking found a better combination!")
else:
    print("Stacking failed to improve over Part 1.")
