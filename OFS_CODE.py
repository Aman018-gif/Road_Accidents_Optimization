import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpInteger, PULP_CBC_CMD
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("rajasthan_accident_data_2018.csv")

# Ensure 'District' column exists
if 'District' not in df.columns:
    raise ValueError("Dataset must include a 'District' column.")

# Create a weighted severity score
df['Weighted_Severity'] = 0.5 * (df['Killed'] / df['Killed'].max()) + \
                          0.3 * (df['Injured'] / df['Injured'].max()) + \
                          0.2 * (df['Accidents'] / df['Accidents'].max())

# Assigning severity weight for both police and awareness campaigns
df['a'] = df['Weighted_Severity']  # For police
df['c'] = df['Weighted_Severity']  # For awareness campaigns

# Define ILP model
model = LpProblem("Rajasthan_Accident_Resource_Optimization", LpMaximize)

# Decision Variables
x = LpVariable.dicts("Police", df.index, lowBound=0, cat=LpInteger)
z = LpVariable.dicts("Awareness", df.index, lowBound=0, cat=LpInteger)

# Cost Parameters (in lakhs)
cost_x = 5   # Cost per police unit deployment (₹5 lakh per unit)
cost_z = 2   # Cost per awareness campaign (₹2 lakh per campaign)
budget = 1800  # Total budget in lakhs (₹18 crore)

# Maximum allocation limits
X_max = 250   # Total police units available
Z_max = 350   # Total awareness campaigns available
District_Limit = 30  # Max allocation per high-risk district (except Jaipur)

# Objective Function - Maximize accident severity reduction
model += lpSum(df.loc[i, 'a'] * x[i] + df.loc[i, 'c'] * z[i] for i in df.index)

# Constraints
# Budget Constraint
model += lpSum(cost_x * x[i] + cost_z * z[i] for i in df.index) <= budget, "Budget_Limit"

# Total available resource constraints
model += lpSum(x[i] for i in df.index) <= X_max, "Max_Police"
model += lpSum(z[i] for i in df.index) <= Z_max, "Max_Awareness"

# Ensure top 5 high-severity districts (except Jaipur) get at least 1 and at most `District_Limit` resources
top5 = df.sort_values(by="Weighted_Severity", ascending=False).head(5).index.tolist()
jaipur_idx = df[df['District'] == 'Jaipur'].index[0]
top5_excluding_jaipur = [i for i in top5 if i != jaipur_idx]
for i in top5_excluding_jaipur:
    model += x[i] + z[i] >= 1, f"Top5_Min_Resource_{i}"
    model += x[i] + z[i] <= District_Limit, f"Top5_Max_Resource_{i}"

# Special rule for Jaipur: Maximize allocation within per-district caps
model += x[jaipur_idx] <= 50, "Jaipur_Max_Police"
model += z[jaipur_idx] <= 40, "Jaipur_Max_Awareness"
model += x[jaipur_idx] >= 40, "Jaipur_Min_Police"  # At least 40 police units
model += z[jaipur_idx] >= 30, "Jaipur_Min_Awareness"  # At least 30 awareness campaigns

# Per-district caps for all districts (including Jaipur)
for i in df.index:
    model += x[i] <= 50, f"Max_Police_Per_District_{i}"
    model += z[i] <= 40, f"Max_Awareness_Per_District_{i}"

# Solve the model
model.solve(PULP_CBC_CMD(msg=0))

# Collect results
df['Police Allocated'] = [x[i].varValue for i in df.index]
df['Awareness Campaigns'] = [z[i].varValue for i in df.index]
df['Total Impact Score'] = df['Police Allocated'] * df['a'] + df['Awareness Campaigns'] * df['c']
df['Normalized Impact (%)'] = 100 * df['Total Impact Score'] / df['Total Impact Score'].max()

# Prepare final output
df_sorted = df[['District', 'Accidents', 'Killed', 'Injured',
                'Police Allocated', 'Awareness Campaigns',
                'Total Impact Score', 'Normalized Impact (%)']]\
                .sort_values(by='Total Impact Score', ascending=False)

# Save results to CSV
df_sorted.to_csv("optimized_allocation.csv", index=False)

# Show summary
print("=== Allocation Summary ===")
print(f"Total Police Allocated: {df['Police Allocated'].sum()}")
print(f"Total Awareness Campaigns: {df['Awareness Campaigns'].sum()}")
print(f"Total Budget Used: {cost_x * df['Police Allocated'].sum() + cost_z * df['Awareness Campaigns'].sum()} lakh INR")

# Show Top 10 districts
print("\n=== Top 10 Districts by Impact ===")
print(df_sorted.head(10))

# Visualization
top10_df = df.sort_values(by='Total Impact Score', ascending=False).head(10)  # Fixed column name

plt.figure(figsize=(14, 6))
sns.barplot(x='District', y='Normalized Impact (%)', data=top10_df, palette='Blues_d')
plt.title('Top 10 Districts by Normalized Impact (%)', fontsize=16)
plt.xlabel('District', fontsize=12)
plt.ylabel('Impact (Normalized %)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Stacked bar for resource allocation
plt.figure(figsize=(14, 6))
bar1 = plt.bar(top10_df['District'], top10_df['Police Allocated'], label='Police', color='steelblue')
bar2 = plt.bar(top10_df['District'], top10_df['Awareness Campaigns'], 
               bottom=top10_df['Police Allocated'], label='Awareness Campaigns', color='lightgreen')
plt.title('Resource Allocation in Top 10 Districts', fontsize=16)
plt.xlabel('District', fontsize=12)
plt.ylabel('Number of Resources', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()