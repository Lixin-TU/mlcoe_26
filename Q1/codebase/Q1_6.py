import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ==========================================
# 1. Setup & Data Generation
# ==========================================

FIRMS = ['Tencent', 'JPMorgan', 'Alibaba_HK', 'Exxon', 'Volkswagen', 'Microsoft', 'Google']

ASSET_COMPONENTS = ['Cash_Equiv', 'Acct_Rec', 'Net_PPE', 'Avail_Sale_Sec', 'Other_Assets']
LIAB_COMPONENTS = ['Curr_Debt', 'Long_Term_Debt', 'Payables', 'Def_Tax_Liab', 'Other_Liab']
EQUITY_COMPONENTS = ['Capital_Stock', 'Common_Stock', 'Ret_Earnings', 'Other_Equity']

BOTTOM_LEVEL_NAMES = ASSET_COMPONENTS + LIAB_COMPONENTS + EQUITY_COMPONENTS

# --- FIX: Define all dimension variables explicitly ---
n_assets = len(ASSET_COMPONENTS) # 5
n_liabs = len(LIAB_COMPONENTS)   # 5
n_equity = len(EQUITY_COMPONENTS) # 4 <--- 补上了这行定义
n_firms = len(FIRMS)
n_bottom_features = len(BOTTOM_LEVEL_NAMES)

# Retained Earnings Index Helper
# Index = 5 (Assets) + 5 (Liabs) + 2 (Capital+Common) = 12
RE_INDEX = n_assets + n_liabs + 2 

print(f"Structure: {n_firms} Firms, {n_bottom_features} Components.")
print(f"Retained Earnings is at Index: {RE_INDEX} ('{BOTTOM_LEVEL_NAMES[RE_INDEX]}')")

def generate_balanced_data():
    """
    Generates synthetic data that strictly respects A = L + E.
    """
    np.random.seed(42)
    # 1. Generate Liabilities and Equity randomly
    # Shape: (n_firms, 5 years, n_liabs + n_equity)
    L_E_data = np.random.uniform(10, 50, size=(n_firms, 5, n_liabs + n_equity))
    
    # 2. Calculate Total L + E
    # Sum across the feature axis (axis 2)
    Total_LE = np.sum(L_E_data, axis=2)
    
    # 3. Generate Asset Components that SUM up to Total_LE
    raw_asset_weights = np.random.uniform(0.5, 1.5, size=(n_firms, 5, n_assets))
    weight_sums = np.sum(raw_asset_weights, axis=2, keepdims=True)
    normalized_weights = raw_asset_weights / weight_sums
    
    # Distribute Total_LE into Asset components
    A_data = normalized_weights * Total_LE[:, :, np.newaxis]
    
    # 4. Concatenate: [Assets, Liabs, Equity]
    data = np.concatenate([A_data, L_E_data], axis=2)
    return data.astype(np.float32)

raw_data_tensor = generate_balanced_data()

# Verification
sample = raw_data_tensor[0, 0, :]
gap = np.sum(sample[:n_assets]) - np.sum(sample[n_assets:])
print(f"Data Generation Gap Check: {gap:.10f}") # Should be near 0

# Constraint Vector for Identity Layer
constraint_vector = np.zeros((n_bottom_features, 1), dtype=np.float32)
constraint_vector[:n_assets] = 1.0
constraint_vector[n_assets:] = -1.0 
constraint_vector_tf = tf.constant(constraint_vector)

# ==========================================
# 2. Model Definition
# ==========================================

class BalanceSheetConstraintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BalanceSheetConstraintLayer, self).__init__(**kwargs)
        self.constraint_vector = constraint_vector_tf

    def call(self, inputs):
        # inputs: Predicted Deltas
        gap = tf.matmul(inputs, self.constraint_vector)
        adjustment = gap / float(n_bottom_features)
        adjustment_vector = tf.matmul(adjustment, tf.transpose(self.constraint_vector))
        return inputs - adjustment_vector

def build_forecasting_model(n_features):
    inputs = layers.Input(shape=(n_features,))
    
    # Simple MLP with L2 Regularization (to handle small data)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    
    # Predict DELTA (Change in balance sheet)
    raw_delta = layers.Dense(n_features, name='raw_delta')(x)
    
    # Apply Horizontal Constraint (A - L - E = 0)
    balanced_delta = BalanceSheetConstraintLayer(name='balanced_delta')(raw_delta)
    
    # Output is the predicted CHANGE
    return Model(inputs=inputs, outputs=balanced_delta)

# ==========================================
# 3. Train the Model (Learning the Deltas)
# ==========================================

model = build_forecasting_model(n_bottom_features)
model.compile(optimizer='adam', loss='mse')

# Prepare Data: X = State(t), Y = State(t+1) - State(t)
X_train_list, Y_delta_list = [], []
for f in range(n_firms):
    for t in range(4): # Years 0->1, 1->2, 2->3, 3->4
        state_t = raw_data_tensor[f, t, :]
        state_next = raw_data_tensor[f, t+1, :]
        delta = state_next - state_t
        
        X_train_list.append(state_t)
        Y_delta_list.append(delta)

X_train = np.array(X_train_list)
Y_train_delta = np.array(Y_delta_list)

print("Training Model to predict Balance Sheet Changes...")
model.fit(X_train, Y_train_delta, epochs=100, batch_size=4, verbose=0)
print("Training Complete.")

# ==========================================
# 4. Forecasting Earnings
# ==========================================

print("\n--- Forecasting Implied Earnings for Year 5 ---")

# Input: Year 4 Balance Sheet
X_test = raw_data_tensor[:, -2, :] # Year 4

# Predict the Change (Delta) for Year 5
predicted_deltas = model.predict(X_test)

# Logic: Net_Income = Delta_RE + Dividends
ASSUMED_DIVIDENDS = 5.0 

for i, firm_name in enumerate(FIRMS):
    # Get predicted change in Retained Earnings
    delta_re = predicted_deltas[i, RE_INDEX]
    
    # Calculate Implied Net Income
    implied_net_income = delta_re + ASSUMED_DIVIDENDS
    
    # Context: Total Asset Growth
    asset_growth = np.sum(predicted_deltas[i, :n_assets])
    
    print(f"Firm: {firm_name:<12}")
    print(f"  > Forecasted Delta RE:  {delta_re:.4f}")
    print(f"  > Assumed Dividends:    {ASSUMED_DIVIDENDS:.4f}")
    print(f"  > IMPLIED NET INCOME:   {implied_net_income:.4f}")
    print(f"  > (Asset Growth:        {asset_growth:.4f})")
    print("-" * 30)