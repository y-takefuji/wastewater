import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('CDC_Wastewater_Data_for_SARS-CoV-2_20250904.csv')

print(f"Original dataset shape: {df.shape}")

if 'pcr_target' in df.columns:
    df = df.drop('pcr_target', axis=1)

target_name = 'pcr_target_flowpop_lin'
orig_nan_count = df[target_name].isna().sum()
orig_effective_count = df.shape[0] - orig_nan_count

print(f"\nTarget variable: {target_name}")
print(f"Original data - Total rows: {df.shape[0]}")
print(f"Original data - Effective values: {orig_effective_count}")
print(f"Original data - NaN values: {orig_nan_count}")
print(f"Original data - NaN percentage: {(orig_nan_count/df.shape[0])*100:.2f}%")

print(f"\nTarget variable data type: {df[target_name].dtype}")
print(f"Target variable sample values: {df[target_name].dropna().head(5).tolist()}")

if df[target_name].dtype == 'object':
    print("\nConverting target variable - removing commas and converting to float...")
    df[target_name] = df[target_name].astype(str).str.replace(',', '').astype(float)

post_conv_nan_count = df[target_name].isna().sum()
post_conv_effective_count = df.shape[0] - post_conv_nan_count

print(f"\nAfter conversion - Effective values: {post_conv_effective_count}")
print(f"After conversion - NaN values: {post_conv_nan_count}")
print(f"After conversion - NaN percentage: {(post_conv_nan_count/df.shape[0])*100:.2f}%")

df = df.dropna(subset=[target_name])
print(f"\nDataset shape after removing NaN targets: {df.shape}")
print(f"Final effective values for target: {df.shape[0]}")

print("\nStatistical summary of target variable (before log transformation):")
print(df[target_name].describe())

print("\nApplying log transformation to target variable...")
epsilon = 1.0
df[target_name + '_log'] = np.log1p(df[target_name] + epsilon)

print("\nStatistical summary of log-transformed target variable:")
print(df[target_name + '_log'].describe())

y = df[target_name + '_log']
X = df.drop([target_name, target_name + '_log'], axis=1)

feature_nan_counts = X.isna().sum()
print("\nFeatures with NaN values:")
print(feature_nan_counts[feature_nan_counts > 0])
print(f"Total features with NaNs: {sum(feature_nan_counts > 0)}")

for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = X[col].astype(str).str.replace(',', '').astype(float)
        except:
            pass

    if X[col].isna().any():
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")

    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

print("Data preprocessing completed.")

print("\nCreating 1/100 reduced dataset by random sampling...")
sample_size = max(int(X.shape[0] * 0.01), 1)
np.random.seed(42)
sampled_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
X_reduced = X.iloc[sampled_indices].copy()
y_reduced = y.iloc[sampled_indices].copy()

print(f"Original dataset: {X.shape[0]} samples")
print(f"Reduced dataset: {X_reduced.shape[0]} samples ({X_reduced.shape[0]/X.shape[0]*100:.2f}%)")

results = {
    'Method': [],
    'CV5': [],
    'CV4': [],
    'Top5 Features': [],
    'Top4 Features': []
}

# ─────────────────────────────────────────────
# Helper: FA global feature selection
# ─────────────────────────────────────────────
def fa_global_feature_selection(X_data, n_top):
    """
    Select top features GLOBALLY across all clusters using Feature Agglomeration.

    Strategy
    --------
    1. Fit FeatureAgglomeration → every feature gets a cluster label.
    2. For each cluster, identify the single best representative feature
       (highest variance within that cluster = most informative member).
    3. Rank those per-cluster representatives by their variance globally
       across all clusters.
    4. Return the top-n feature names from that global ranking.

    This guarantees:
    - At most one feature per cluster (diversity / no redundancy).
    - Global ranking, not per-cluster independent selection.
    """
    n_clusters = min(int(X_data.shape[1] * 0.2), 20, X_data.shape[1])
    n_clusters = max(n_clusters, n_top)          # need at least n_top clusters

    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_data)

    feature_variances = X_data.var().values       # shape: (n_features,)
    labels = fa.labels_                           # shape: (n_features,)

    # Step A: pick the best representative feature from each cluster
    cluster_best = {}          # cluster_id → (feature_idx, variance)
    for feat_idx, (clust_id, var) in enumerate(zip(labels, feature_variances)):
        if clust_id not in cluster_best or var > cluster_best[clust_id][1]:
            cluster_best[clust_id] = (feat_idx, var)

    # Step B: collect representative (feature_idx, variance) pairs
    representatives = list(cluster_best.values())   # one entry per cluster

    # Step C: global ranking — sort all representatives by variance descending
    representatives.sort(key=lambda x: x[1], reverse=True)

    # Step D: take top-n
    top_indices = [feat_idx for feat_idx, _ in representatives[:n_top]]
    top_features = X_data.columns[top_indices].tolist()
    return top_features


# ─────────────────────────────────────────────
# 1. Random Forest Feature Selection
# ─────────────────────────────────────────────
print("\nRunning Random Forest Feature Selection...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_reduced, y_reduced)
feature_importances = rf.feature_importances_
rf_features_idx = np.argsort(feature_importances)[::-1]
top5_rf_features = X_reduced.columns[rf_features_idx[:5]].tolist()

X_cv5_rf = X_reduced[top5_rf_features]
cv5_rf_score = cross_val_score(rf, X_cv5_rf, y_reduced, cv=5, scoring='r2').mean()

reduced_fullset = X_reduced.drop(top5_rf_features[0], axis=1)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(reduced_fullset, y_reduced)
feature_importances2 = rf2.feature_importances_
rf_features_idx2 = np.argsort(feature_importances2)[::-1]
top4_rf_features_reduced = reduced_fullset.columns[rf_features_idx2[:4]].tolist()

X_cv4_rf = reduced_fullset[top4_rf_features_reduced]
cv4_rf_score = cross_val_score(rf2, X_cv4_rf, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Random Forest')
results['CV5'].append(cv5_rf_score)
results['CV4'].append(cv4_rf_score)
results['Top5 Features'].append(', '.join(top5_rf_features))
results['Top4 Features'].append(', '.join(top4_rf_features_reduced))

# ─────────────────────────────────────────────
# 2. XGBoost Feature Selection
# ─────────────────────────────────────────────
print("Running XGBoost Feature Selection...")
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_reduced, y_reduced)
feature_importances = xgb.feature_importances_
xgb_features_idx = np.argsort(feature_importances)[::-1]
top5_xgb_features = X_reduced.columns[xgb_features_idx[:5]].tolist()

X_cv5_xgb = X_reduced[top5_xgb_features]
cv5_xgb_score = cross_val_score(xgb, X_cv5_xgb, y_reduced, cv=5, scoring='r2').mean()

reduced_fullset = X_reduced.drop(top5_xgb_features[0], axis=1)
xgb2 = XGBRegressor(n_estimators=100, random_state=42)
xgb2.fit(reduced_fullset, y_reduced)
feature_importances2 = xgb2.feature_importances_
xgb_features_idx2 = np.argsort(feature_importances2)[::-1]
top4_xgb_features_reduced = reduced_fullset.columns[xgb_features_idx2[:4]].tolist()

X_cv4_xgb = reduced_fullset[top4_xgb_features_reduced]
cv4_xgb_score = cross_val_score(xgb2, X_cv4_xgb, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('XGBoost')
results['CV5'].append(cv5_xgb_score)
results['CV4'].append(cv4_xgb_score)
results['Top5 Features'].append(', '.join(top5_xgb_features))
results['Top4 Features'].append(', '.join(top4_xgb_features_reduced))

# ─────────────────────────────────────────────
# 3. Feature Agglomeration (FA) — FIXED
#    Global selection across all clusters
# ─────────────────────────────────────────────
print("Running Feature Agglomeration (FA) with global feature selection...")

# CV5: top 5 features globally
top5_fa_features = fa_global_feature_selection(X_reduced, n_top=5)
print(f"  FA top-5 features (global): {top5_fa_features}")

X_cv5_fa = X_reduced[top5_fa_features]
regressor_fa = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_fa_score = cross_val_score(regressor_fa, X_cv5_fa, y_reduced, cv=5, scoring='r2').mean()

# CV4: remove highest feature, re-run global FA on reduced set
reduced_fullset = X_reduced.drop(top5_fa_features[0], axis=1)
top4_fa_features_reduced = fa_global_feature_selection(reduced_fullset, n_top=4)
print(f"  FA top-4 features after drop (global): {top4_fa_features_reduced}")

X_cv4_fa = reduced_fullset[top4_fa_features_reduced]
regressor_fa2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_fa_score = cross_val_score(regressor_fa2, X_cv4_fa, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Feature Agglomeration')
results['CV5'].append(cv5_fa_score)
results['CV4'].append(cv4_fa_score)
results['Top5 Features'].append(', '.join(top5_fa_features))
results['Top4 Features'].append(', '.join(top4_fa_features_reduced))

# ─────────────────────────────────────────────
# 4. Highly Variable Gene Selection (HVGS)
# ─────────────────────────────────────────────
print("Running Highly Variable Gene Selection (HVGS)...")
feature_variance = X_reduced.var(axis=0)
hvgs_features_idx = np.argsort(feature_variance)[::-1]
top5_hvgs_features = X_reduced.columns[hvgs_features_idx[:5]].tolist()

X_cv5_hvgs = X_reduced[top5_hvgs_features]
regressor_hvgs = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_hvgs_score = cross_val_score(regressor_hvgs, X_cv5_hvgs, y_reduced, cv=5, scoring='r2').mean()

reduced_fullset = X_reduced.drop(top5_hvgs_features[0], axis=1)
feature_variance_reduced = reduced_fullset.var(axis=0)
hvgs_features_idx_reduced = np.argsort(feature_variance_reduced)[::-1]
top4_hvgs_features_reduced = reduced_fullset.columns[hvgs_features_idx_reduced[:4]].tolist()

X_cv4_hvgs = reduced_fullset[top4_hvgs_features_reduced]
regressor_hvgs2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_hvgs_score = cross_val_score(regressor_hvgs2, X_cv4_hvgs, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('HVGS')
results['CV5'].append(cv5_hvgs_score)
results['CV4'].append(cv4_hvgs_score)
results['Top5 Features'].append(', '.join(top5_hvgs_features))
results['Top4 Features'].append(', '.join(top4_hvgs_features_reduced))

# ─────────────────────────────────────────────
# 5. Spearman Correlation
# ─────────────────────────────────────────────
print("Running Spearman Correlation Feature Selection...")
spearman_corrs = []
for col in X_reduced.columns:
    corr, _ = spearmanr(X_reduced[col], y_reduced)
    spearman_corrs.append((col, abs(corr)))

spearman_corrs.sort(key=lambda x: x[1], reverse=True)
top5_spearman_features = [item[0] for item in spearman_corrs[:5]]

X_cv5_spearman = X_reduced[top5_spearman_features]
regressor_spearman = RandomForestRegressor(n_estimators=100, random_state=42)
cv5_spearman_score = cross_val_score(regressor_spearman, X_cv5_spearman, y_reduced, cv=5, scoring='r2').mean()

reduced_fullset = X_reduced.drop(top5_spearman_features[0], axis=1)
spearman_corrs_reduced = []
for col in reduced_fullset.columns:
    corr, _ = spearmanr(reduced_fullset[col], y_reduced)
    spearman_corrs_reduced.append((col, abs(corr)))

spearman_corrs_reduced.sort(key=lambda x: x[1], reverse=True)
top4_spearman_features_reduced = [item[0] for item in spearman_corrs_reduced[:4]]

X_cv4_spearman = reduced_fullset[top4_spearman_features_reduced]
regressor_spearman2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv4_spearman_score = cross_val_score(regressor_spearman2, X_cv4_spearman, y_reduced, cv=5, scoring='r2').mean()

results['Method'].append('Spearman')
results['CV5'].append(cv5_spearman_score)
results['CV4'].append(cv4_spearman_score)
results['Top5 Features'].append(', '.join(top5_spearman_features))
results['Top4 Features'].append(', '.join(top4_spearman_features_reduced))

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df['CV5'] = results_df['CV5'].apply(lambda x: f"{x:.4f}")
results_df['CV4'] = results_df['CV4'].apply(lambda x: f"{x:.4f}")

print("\nSummary Results:")
print(results_df)

results_df.to_csv('result.csv', index=False)
print("\nResults saved to 'result.csv'")