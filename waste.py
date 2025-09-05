import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# 1. Load data
fn = "CDC_Wastewater_Data_for_SARS-CoV-2_20250904.csv"
df = pd.read_csv(fn, low_memory=False)

# 2. Drop the raw PCR target column
if 'pcr_target' in df.columns:
    df = df.drop(columns=['pcr_target'])

# 3. Parse dates
df['sample_collect_date'] = pd.to_datetime(df['sample_collect_date'], errors='coerce')

# 4. Filter for data between 2021-2025
df = df[(df['sample_collect_date'] >= '2021-01-01') & (df['sample_collect_date'] <= '2025-12-31')]

# 5. Convert pcr_target_flowpop_lin to numeric, handling commas in numbers
if 'pcr_target_flowpop_lin' in df.columns:
    # Convert to string first to ensure consistent handling
    df['pcr_target_flowpop_lin'] = df['pcr_target_flowpop_lin'].astype(str)
    # Remove commas and convert to float
    df['pcr_target_flowpop_lin'] = df['pcr_target_flowpop_lin'].str.replace(',', '').replace('', np.nan)
    df['pcr_target_flowpop_lin'] = pd.to_numeric(df['pcr_target_flowpop_lin'], errors='coerce')
else:
    print("Warning: 'pcr_target_flowpop_lin' column not found in dataset")

# 6. Filter out any bad rows
df = df.dropna(subset=['sample_collect_date', 'pcr_target_flowpop_lin', 'sewershed_id'])
print('how many sewershed id', df['sewershed_id'].value_counts())

# 7. Find the top 50 sewershed_id plants with the most data points
data_counts = df.groupby('sewershed_id').size().sort_values(ascending=False)
top50_by_count = data_counts.head(50).index.tolist()
print(f"Top 50 sewershed_id by data count (first 5): {top50_by_count[:5]}")

# 8. Filter to only include the top 50 treatment plants
df_top50 = df[df['sewershed_id'].isin(top50_by_count)]

# 9. Compute daily per-capita viral shedding per sewershed
daily = (
    df_top50
    .groupby(['sewershed_id', 'sample_collect_date'])['pcr_target_flowpop_lin']
    .mean()
    .reset_index(name='daily_flowpop_lin')
)

# 10. Compute overall average signal by sewershed among top 50
sewer_means = (
    daily
    .groupby('sewershed_id')['daily_flowpop_lin']
    .mean()
    .reset_index(name='mean_flowpop_lin')
    .sort_values('mean_flowpop_lin', ascending=False)
)

# 11. Get the top 4 sewersheds with highest signal from the top 50
top4 = sewer_means.head(4)['sewershed_id'].tolist()
print("Top 4 sewershed_id by mean per-capita signal (from top 50 with most data):", top4)

# Extract jurisdiction information for the top 4 sewersheds
jurisdiction_mapping = {}
for sid in top4:
    # Get the most common jurisdiction for this sewershed_id
    jurisdiction = df_top50[df_top50['sewershed_id'] == sid]['wwtp_jurisdiction'].value_counts().index[0]
    jurisdiction_mapping[sid] = jurisdiction
    
print("Sewershed IDs and their jurisdictions:")
for sid, jur in jurisdiction_mapping.items():
    print(f"ID {sid}: {jur}")

# 12. Filter the daily dataframe to only those top 4
top4_daily = daily[daily['sewershed_id'].isin(top4)]

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the plotted data as CSV (the top4_daily dataframe)
plotted_data_path = os.path.join(output_dir, "top4_sewersheds_plotted_data.csv")
top4_daily.to_csv(plotted_data_path, index=False)
print(f"Plotted data saved as: {plotted_data_path}")

# 13. Plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 12)

# Define line styles for black and white plotting - all thin
line_styles = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
line_width = 1.0  # thin width for all lines

# Create figure with more space at the top for the title
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve top 5% for the title
# Plot on both axes with different line styles
for i, sid in enumerate(top4):
    subset = top4_daily[top4_daily['sewershed_id'] == sid].sort_values('sample_collect_date')
    jurisdiction = jurisdiction_mapping[sid]
    
    # Select line style
    style_idx = i % len(line_styles)
    
    # Plot on first (zoomed) axis with jurisdiction in the label
    ax1.plot(subset['sample_collect_date'],
             subset['daily_flowpop_lin'],
             label=f"ID: {sid} ({jurisdiction})",
             color='black',
             linestyle=line_styles[style_idx],
             linewidth=line_width)
    
    # Plot on second (full range) axis with jurisdiction in the label
    ax2.plot(subset['sample_collect_date'],
             subset['daily_flowpop_lin'],
             label=f"ID: {sid} ({jurisdiction})",
             color='black',
             linestyle=line_styles[style_idx],
             linewidth=line_width)

# Configure the zoomed axis (0 to 0.25×10¹¹)
ax1.set_ylim(0, 0.25e11)
ax1.set_ylabel("Mean per-capita viral signal\n(pcr_target_flowpop_lin)")
ax1.grid(True)
ax1.legend(title="Sewershed ID", loc='upper right')

# Configure the full range axis
max_val = top4_daily['daily_flowpop_lin'].max() * 1.05  # 5% headroom
ax2.set_ylim(0, max_val)
ax2.set_title("Full Range View")
ax2.set_xlabel("Sample Collection Date")
ax2.set_ylabel("Mean per-capita viral signal\n(pcr_target_flowpop_lin)")
ax2.grid(True)

# Add data count annotation for each sewershed in the second legend
legend_elements = []
for i, sid in enumerate(top4):
    count = data_counts[sid]
    jurisdiction = jurisdiction_mapping[sid]
    style_idx = i % len(line_styles)
    legend_elements.append(plt.Line2D([0], [0], color='black', 
                          linestyle=line_styles[style_idx],
                          linewidth=line_width,
                          label=f"ID {sid} ({jurisdiction}): {count} samples"))
    
ax2.legend(handles=legend_elements, title="Sewershed ID")

# Update the title to reflect 4 sewersheds instead of 5
fig.suptitle("Top 4 Sewersheds with Highest SARS-CoV-2 Signals\nSelected from 50 Treatment Plants with Most Data", 
             fontsize=16, y=0.98)

# Add an annotation indicating the zoom level instead of a title
ax1.annotate('Zoomed view (0 to 0.25×10¹¹)', xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

plt.subplots_adjust(top=0.93, hspace=0.3)  # Increased space between plots

# Save the figure as PNG with high resolution
png_path = os.path.join(output_dir, "top4_sewersheds_covid_signals.png")
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Figure saved as: {png_path} (300 DPI)")

# Display the figure
plt.show()

# 14. Create and save a summary table of the top 4 sewersheds
top4_summary = pd.DataFrame({
    'sewershed_id': top4,
    'jurisdiction': [jurisdiction_mapping[sid] for sid in top4],
    'total_samples': [data_counts[sid] for sid in top4],
    'mean_viral_signal': [sewer_means[sewer_means['sewershed_id']==sid]['mean_flowpop_lin'].iloc[0] for sid in top4],
    'date_range': [f"{daily[daily['sewershed_id']==sid]['sample_collect_date'].min().strftime('%Y-%m-%d')} to {daily[daily['sewershed_id']==sid]['sample_collect_date'].max().strftime('%Y-%m-%d')}" for sid in top4]
})

print("\nSummary of top 4 sewersheds:")
print(top4_summary)

# Save the summary table as CSV
csv_path = os.path.join(output_dir, "top4_sewersheds_summary.csv")
top4_summary.to_csv(csv_path, index=False)
print(f"Summary table saved as: {csv_path}")
