import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances

# Load the dataset
df = pd.read_csv('CDC_Wastewater_Data_for_SARS-CoV-2_20250904.csv')

# Show shape of dataset
print(f"Original dataset shape: {df.shape}")

# Drop 'pcr_target' 
if 'pcr_target' in df.columns:
    df = df.drop('pcr_target', axis=1)

# Target variable information
target_name = 'pcr_target_flowpop_lin'
orig_nan_count = df[target_name].isna().sum()
orig_effective_count = df.shape[0] - orig_nan_count

print(f"\nTarget variable: {target_name}")
print(f"Original data - Total rows: {df.shape[0]}")
print(f"Original data - Effective values: {orig_effective_count}")
print(f"Original data - NaN values: {orig_nan_count}")
print(f"Original data - NaN percentage: {(orig_nan_count/df.shape[0])*100:.2f}%")

# Check data type and handle target variable
print(f"\nTarget variable data type: {df[target_name].dtype}")
print(f"Target variable sample values: {df[target_name].dropna().head(5).tolist()}")

# Simply remove commas and convert to float
if df[target_name].dtype == 'object':
    print("\nConverting target variable - removing commas and converting to float...")
    df[target_name] = df[target_name].astype(str).str.replace(',', '').astype(float)

# Check for NaN values in target variable after conversion
post_conv_nan_count = df[target_name].isna().sum()
post_conv_effective_count = df.shape[0] - post_conv_nan_count

print(f"\nAfter conversion - Effective values: {post_conv_effective_count}")
print(f"After conversion - NaN values: {post_conv_nan_count}")
print(f"After conversion - NaN percentage: {(post_conv_nan_count/df.shape[0])*100:.2f}%")

# Remove rows with NaN in target variable
df = df.dropna(subset=[target_name])
print(f"\nDataset shape after removing NaN targets: {df.shape}")
print(f"Final effective values for target: {df.shape[0]}")

# Display statistical summary of target variable before log transformation
print("\nStatistical summary of target variable (before log transformation):")
print(df[target_name].describe())

# Apply log transformation to target variable
print("\nApplying log transformation to target variable...")
epsilon = 1.0  # Small constant to add before log to handle zeros or very small values
df[target_name + '_log'] = np.log1p(df[target_name] + epsilon)

# Display statistical summary of log-transformed target variable
print("\nStatistical summary of log-transformed target variable:")
print(df[target_name + '_log'].describe())

# Set new log-transformed target variable
y = df[target_name + '_log']
X = df.drop([target_name, target_name + '_log'], axis=1)

# Check for NaN values in features
feature_nan_counts = X.isna().sum()
print("\nFeatures with NaN values:")
print(feature_nan_counts[feature_nan_counts > 0])
print(f"Total features with NaNs: {sum(feature_nan_counts > 0)}")

# Handle mixed type columns properly
for col in X.columns:
    # Check if column contains numeric values but as strings with commas
    if X[col].dtype == 'object':
        # Try to convert string numbers with commas to float
        try:
            X[col] = X[col].astype(str).str.replace(',', '').astype(float)
        except:
            pass  # Keep as is if conversion fails
    
    # Check column types and handle NaNs
    if X[col].isna().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            # For numeric columns, fill NaN with median
            X[col] = X[col].fillna(X[col].median())
        else:
            # For non-numeric columns, fill NaN with a placeholder
            X[col] = X[col].fillna("missing")
    
    # Label encode any remaining string columns
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

print("Data preprocessing completed.")

# Create a 1/100 reduced dataset by random sampling
print("\nCreating 1/100 reduced dataset by random sampling...")
# Calculate sample size (1%)
sample_size = max(int(X.shape[0] * 0.01), 1)  # Ensure at least 1 sample
# Random sampling without replacement
np.random.seed(42)  # Set seed for reproducibility
sampled_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
X_reduced = X.iloc[sampled_indices].copy()
y_reduced = y.iloc[sampled_indices].copy()

print(f"Original dataset: {X.shape[0]} samples")
print(f"Reduced dataset: {X_reduced.shape[0]} samples ({X_reduced.shape[0]/X.shape[0]*100:.2f}%)")

# Initialize results dictionary
results = {
    'Method': [],
    'CV5': [],
    'CV4': [],
    'Top5 Features': [],
    'Top4 Features': []
}

# 1. Random Forest Feature Selection
print("\nRunning Random Forest Feature Selection...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_reduced, y_reduced)
feature_importances = rf.feature_importances_
rf_features_idx = np.argsort(feature_importances)[::-1]
top5_rf_features = X_reduced.columns[rf_features_idx[:5]].tolist()

# Create CV5 dataset with top 5 features
X_cv5_rf = X_reduced[top5_rf_features]
cv5_rf_score = cross_val_score(rf, X_cv5_rf, y_reduced, cv=5, scoring='r2').mean()

# Create reduced dataset by removing the highest feature
reduced_fullset = X_reduced.drop(top5_rf_features[0], axis=1)
# Re-run feature selection on reduced dataset
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(reduced_fullset, y_reduced)
feature_importances2 = rf2.feature_importances_
rf_features_idx2 = np.argsort(feature_importances2)[::-1]
top4_rf_features_reduced = reduced_fullset.columns[rf_features_idx2[:4]].tolist()

# Create CV4 dataset with top 4 features from reduced dataset
X_cv4_rf = reduced_fullset[top4_rf_features_reduced]
cv4_rf_score = cross_val_score(rf2, X_cv4_rf, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Random Forest')
results['CV5'].append(cv5_rf_score)
results['CV4'].append(cv4_rf_score)
results['Top5 Features'].append(', '.join(top5_rf_features))
results['Top4 Features'].append(', '.join(top4_rf_features_reduced))

# 2. XGBoost Feature Selection
print("Running XGBoost Feature Selection...")
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_reduced, y_reduced)
feature_importances = xgb.feature_importances_
xgb_features_idx = np.argsort(feature_importances)[::-1]
top5_xgb_features = X_reduced.columns[xgb_features_idx[:5]].tolist()

# Create CV5 dataset with top 5 features
X_cv5_xgb = X_reduced[top5_xgb_features]
cv5_xgb_score = cross_val_score(xgb, X_cv5_xgb, y_reduced, cv=5, scoring='r2').mean()

# Create reduced dataset by removing the highest feature
reduced_fullset = X_reduced.drop(top5_xgb_features[0], axis=1)
# Re-run feature selection on reduced dataset
xgb2 = XGBRegressor(n_estimators=100, random_state=42)
xgb2.fit(reduced_fullset, y_reduced)
feature_importances2 = xgb2.feature_importances_
xgb_features_idx2 = np.argsort(feature_importances2)[::-1]
top4_xgb_features_reduced = reduced_fullset.columns[xgb_features_idx2[:4]].tolist()

# Create CV4 dataset with top 4 features from reduced dataset
X_cv4_xgb = reduced_fullset[top4_xgb_features_reduced]
cv4_xgb_score = cross_val_score(xgb2, X_cv4_xgb, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('XGBoost')
results['CV5'].append(cv5_xgb_score)
results['CV4'].append(cv4_xgb_score)
results['Top5 Features'].append(', '.join(top5_xgb_features))
results['Top4 Features'].append(', '.join(top4_xgb_features_reduced))

# 3. Feature Agglomeration (FA) - Simplified
print("Running Feature Agglomeration (FA)...")
# Determine number of clusters to create
n_clusters = min(int(X_reduced.shape[1] * 0.2), 20)  # 20% of features, max 20 clusters
fa = FeatureAgglomeration(n_clusters=n_clusters)
fa.fit(X_reduced)

# Step 1: Calculate variance of each feature
feature_variances = X_reduced.var().values
norm_var = feature_variances / np.max(feature_variances) if np.max(feature_variances) > 0 else feature_variances

# Step 2: Calculate cluster distances
# For each feature, calculate the average distance between its cluster and all other clusters
cluster_distances = np.zeros(X_reduced.shape[1])

# Create transformed data for distance calculations
X_transformed = fa.transform(X_reduced)

# Calculate distance between clusters
for i in range(X_reduced.shape[1]):
    # Get the cluster this feature belongs to
    cluster_i = fa.labels_[i]
    
    # Calculate average distance from this cluster to all others
    distances = []
    for j in range(n_clusters):
        if j != cluster_i:
            # Get the mean vector for each cluster
            cluster_i_data = X_transformed[:, cluster_i].reshape(-1, 1)
            cluster_j_data = X_transformed[:, j].reshape(-1, 1)
            
            # Calculate Euclidean distance between cluster means
            dist = np.mean(np.sqrt(np.sum((cluster_i_data - cluster_j_data) ** 2, axis=1)))
            distances.append(dist)
    
    if distances:
        cluster_distances[i] = np.mean(distances)

# Normalize cluster distances
norm_dist = cluster_distances / np.max(cluster_distances) if np.max(cluster_distances) > 0 else cluster_distances

# Step 3: Combine metrics (90% variance, 10% distance)
feature_importance = 0.9 * norm_var + 0.1 * norm_dist

# Step 4: Sort features by importance
fa_features_idx = np.argsort(feature_importance)[::-1]
top5_fa_features = X_reduced.columns[fa_features_idx[:5]].tolist()

# Create CV5 dataset
X_cv5_fa = X_reduced[top5_fa_features]
regressor_fa = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_fa_score = cross_val_score(regressor_fa, X_cv5_fa, y_reduced, cv=5, scoring='r2').mean()

# Create reduced dataset by removing highest feature
reduced_fullset = X_reduced.drop(top5_fa_features[0], axis=1)

# Re-run Feature Agglomeration on reduced dataset
n_clusters_reduced = min(int(reduced_fullset.shape[1] * 0.2), 20)
fa_reduced = FeatureAgglomeration(n_clusters=n_clusters_reduced)
fa_reduced.fit(reduced_fullset)

# Calculate feature importance for reduced dataset
feature_variances_reduced = reduced_fullset.var().values
norm_var_reduced = feature_variances_reduced / np.max(feature_variances_reduced) if np.max(feature_variances_reduced) > 0 else feature_variances_reduced

# Calculate cluster distances for reduced dataset
cluster_distances_reduced = np.zeros(reduced_fullset.shape[1])

# Create transformed data for distance calculations
X_transformed_reduced = fa_reduced.transform(reduced_fullset)

# Calculate distance between clusters for reduced dataset
for i in range(reduced_fullset.shape[1]):
    # Get the cluster this feature belongs to
    cluster_i = fa_reduced.labels_[i]
    
    # Calculate average distance from this cluster to all others
    distances = []
    for j in range(n_clusters_reduced):
        if j != cluster_i:
            # Get the mean vector for each cluster
            cluster_i_data = X_transformed_reduced[:, cluster_i].reshape(-1, 1)
            cluster_j_data = X_transformed_reduced[:, j].reshape(-1, 1)
            
            # Calculate Euclidean distance between cluster means
            dist = np.mean(np.sqrt(np.sum((cluster_i_data - cluster_j_data) ** 2, axis=1)))
            distances.append(dist)
    
    if distances:
        cluster_distances_reduced[i] = np.mean(distances)

# Normalize cluster distances for reduced dataset
norm_dist_reduced = cluster_distances_reduced / np.max(cluster_distances_reduced) if np.max(cluster_distances_reduced) > 0 else cluster_distances_reduced

# Combine metrics for reduced dataset
feature_importance_reduced = 0.9 * norm_var_reduced + 0.1 * norm_dist_reduced

# Sort features by importance
fa_features_idx_reduced = np.argsort(feature_importance_reduced)[::-1]
top4_fa_features_reduced = reduced_fullset.columns[fa_features_idx_reduced[:4]].tolist()

# Create CV4 dataset
X_cv4_fa = reduced_fullset[top4_fa_features_reduced]
regressor_fa2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_fa_score = cross_val_score(regressor_fa2, X_cv4_fa, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Feature Agglomeration')
results['CV5'].append(cv5_fa_score)
results['CV4'].append(cv4_fa_score)
results['Top5 Features'].append(', '.join(top5_fa_features))
results['Top4 Features'].append(', '.join(top4_fa_features_reduced))

# 4. Highly Variable Gene Selection (HVGS)
print("Running Highly Variable Gene Selection (HVGS)...")
# Calculate variance of each feature
feature_variance = X_reduced.var(axis=0)
# Sort features by variance
hvgs_features_idx = np.argsort(feature_variance)[::-1]
top5_hvgs_features = X_reduced.columns[hvgs_features_idx[:5]].tolist()

# Create CV5 dataset with top 5 features
X_cv5_hvgs = X_reduced[top5_hvgs_features]
regressor_hvgs = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_hvgs_score = cross_val_score(regressor_hvgs, X_cv5_hvgs, y_reduced, cv=5, scoring='r2').mean()

# Create reduced dataset by removing the highest feature
reduced_fullset = X_reduced.drop(top5_hvgs_features[0], axis=1)
# Recalculate variance on reduced dataset
feature_variance_reduced = reduced_fullset.var(axis=0)
hvgs_features_idx_reduced = np.argsort(feature_variance_reduced)[::-1]
top4_hvgs_features_reduced = reduced_fullset.columns[hvgs_features_idx_reduced[:4]].tolist()

# Create CV4 dataset with top 4 features from reduced dataset
X_cv4_hvgs = reduced_fullset[top4_hvgs_features_reduced]
regressor_hvgs2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_hvgs_score = cross_val_score(regressor_hvgs2, X_cv4_hvgs, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('HVGS')
results['CV5'].append(cv5_hvgs_score)
results['CV4'].append(cv4_hvgs_score)
results['Top5 Features'].append(', '.join(top5_hvgs_features))
results['Top4 Features'].append(', '.join(top4_hvgs_features_reduced))

# 5. Spearman Correlation
print("Running Spearman Correlation Feature Selection...")
spearman_corrs = []
for col in X_reduced.columns:
    corr, _ = spearmanr(X_reduced[col], y_reduced)
    spearman_corrs.append((col, abs(corr)))  # Use absolute correlation

# Sort by absolute correlation value
spearman_corrs.sort(key=lambda x: x[1], reverse=True)
top5_spearman_features = [item[0] for item in spearman_corrs[:5]]

# Create CV5 dataset with top 5 features
X_cv5_spearman = X_reduced[top5_spearman_features]
regressor_spearman = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_spearman_score = cross_val_score(regressor_spearman, X_cv5_spearman, y_reduced, cv=5, scoring='r2').mean()

# Create reduced dataset by removing the highest feature
reduced_fullset = X_reduced.drop(top5_spearman_features[0], axis=1)
# Recalculate Spearman correlations on reduced dataset
spearman_corrs_reduced = []
for col in reduced_fullset.columns:
    corr, _ = spearmanr(reduced_fullset[col], y_reduced)
    spearman_corrs_reduced.append((col, abs(corr)))

spearman_corrs_reduced.sort(key=lambda x: x[1], reverse=True)
top4_spearman_features_reduced = [item[0] for item in spearman_corrs_reduced[:4]]

# Create CV4 dataset with top 4 features from reduced dataset
X_cv4_spearman = reduced_fullset[top4_spearman_features_reduced]
regressor_spearman2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_spearman_score = cross_val_score(regressor_spearman2, X_cv4_spearman, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Spearman')
results['CV5'].append(cv5_spearman_score)
results['CV4'].append(cv4_spearman_score)
results['Top5 Features'].append(', '.join(top5_spearman_features))
results['Top4 Features'].append(', '.join(top4_spearman_features_reduced))

# Create summary table with 4 decimal places
results_df = pd.DataFrame(results)
# Format CV scores to 4 decimal places
results_df['CV5'] = results_df['CV5'].apply(lambda x: f"{x:.4f}")
results_df['CV4'] = results_df['CV4'].apply(lambda x: f"{x:.4f}")

print("\nSummary Results:")
print(results_df)

# Save the results table to CSV
results_df.to_csv('result.csv', index=False)
print("\nResults saved to 'result.csv'")
