import pandas as pd
import yaml
import json
import os

# Define the directory path
directory_path = r'C:\Users\USER\Desktop\Onchain Labs\Retro Funding'

# Load YAML configuration
yaml_path = os.path.join(directory_path, 'heuristics.yaml')
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Missing YAML file: {yaml_path}")

with open(yaml_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Extract weights, thresholds, and file names
weights = config['model']['parameters']['weights']
thresholds = config['model']['parameters']['thresholds']
projects_file = config['data']['projects_file']
contributor_file = 'devtooling_dependency_graph.csv'  # Default to dependency graph
onchain_metrics_file = 'onchain_project_metadata.csv'  # Use alternative file
dependency_file = config['data']['dependency_file']

# Construct file paths
projects_path = os.path.join(directory_path, projects_file)
contributor_path = os.path.join(directory_path, contributor_file)
onchain_metrics_path = os.path.join(directory_path, onchain_metrics_file)
dependency_path = os.path.join(directory_path, dependency_file)

# Verify required files
required_files = {'projects_file': projects_path, 'onchain_metrics_file': onchain_metrics_path, 'dependency_file': dependency_path}
for file_name, file_path in required_files.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

# Load data
try:
    projects_df = pd.read_csv(projects_path, encoding='utf-8')
except UnicodeDecodeError:
    print("Warning: UTF-8 encoding failed for projects CSV. Trying 'latin1'.")
    projects_df = pd.read_csv(projects_path, encoding='latin1')
if 'project_id' not in projects_df.columns:
    raise KeyError("Column 'project_id' not found in devtooling_project_metadata.csv")

try:
    onchain_df = pd.read_csv(onchain_metrics_path, encoding='utf-8')
except UnicodeDecodeError:
    print("Warning: UTF-8 encoding failed for onchain CSV. Trying 'latin1'.")
    onchain_df = pd.read_csv(onchain_metrics_path, encoding='latin1')

try:
    with open(dependency_path, 'r', encoding='utf-8') as file:
        raw_metrics = json.load(file)
    dependents_df = pd.DataFrame(raw_metrics)
except UnicodeDecodeError:
    print("Warning: UTF-8 encoding failed for JSON file. Trying 'latin1'.")
    with open(dependency_path, 'r', encoding='latin1') as file:
        raw_metrics = json.load(file)
    dependents_df = pd.DataFrame(raw_metrics)

# Handle contributor data
try:
    contributor_df = pd.read_csv(contributor_path, encoding='utf-8')
    # Rename project_id if needed
    possible_id_cols = ['project_id', 'ProjectID', 'projectid', 'ProjectId']
    project_id_col = next((col for col in possible_id_cols if col in contributor_df.columns), None)
    if project_id_col:
        contributor_df = contributor_df.rename(columns={project_id_col: 'project_id'})
    else:
        contributor_df['project_id'] = projects_df['project_id']
    # Rename developer_id
    possible_dev_cols = ['developer_id', 'contributor_count', 'developers']
    dev_col = next((col for col in possible_dev_cols if col in contributor_df.columns), None)
    if dev_col:
        contributor_df = contributor_df.rename(columns={dev_col: 'developer_id'})
    else:
        contributor_df['developer_id'] = 0
except FileNotFoundError:
    print("Warning: 'devtooling_dependency_graph.csv' not found. Setting 'developer_id' to 0.")
    contributor_df = pd.DataFrame({'project_id': projects_df['project_id'], 'developer_id': 0})
except UnicodeDecodeError:
    print("Warning: UTF-8 encoding failed for dependency graph CSV. Trying 'latin1'.")
    contributor_df = pd.read_csv(contributor_path, encoding='latin1')

# Set developer_connection_count to 0 if missing
if 'developer_connection_count' not in dependents_df.columns:
    dependents_df['developer_connection_count'] = 0

# Use developer_id from raw_metrics if available
if 'developer_id' in dependents_df.columns:
    contributor_df = dependents_df[['project_id', 'developer_id']]

# Merge data
merged_df = projects_df[['project_id', 'star_count', 'fork_count']]
merged_df = merged_df.merge(contributor_df[['project_id', 'developer_id']], on='project_id', how='left')
merged_df = merged_df.merge(dependents_df[['project_id', 'developer_connection_count']], on='project_id', how='left')
onchain_cols = ['project_id']
for col in ['total_transaction_count', 'total_active_addresses_count', 'total_gas_fees']:
    if col in onchain_df.columns:
        onchain_cols.append(col)
    else:
        merged_df[col] = 0
merged_df = merged_df.merge(onchain_df[onchain_cols], on='project_id', how='left')

# Rename columns to match YAML
merged_df = merged_df.rename(columns={
    'star_count': 'stars',
    'fork_count': 'forks',
    'developer_id': 'contributors',
    'developer_connection_count': 'dependents',
    'total_transaction_count': 'tx_count',
    'total_active_addresses_count': 'unique_users',
    'total_gas_fees': 'gas_fee'
})

# Filter projects (thresholds set to 0)
eligible_df = merged_df[
    (merged_df['dependents'].fillna(0) >= 0) &
    (merged_df['contributors'].fillna(0) >= 0)
]

# Normalize metrics
metrics = ['stars', 'forks', 'contributors', 'dependents', 'tx_count', 'unique_users', 'gas_fee']
for metric in metrics:
    if metric not in merged_df.columns:
        eligible_df[metric] = 0
        eligible_df[f'normalized_{metric}'] = 0
        continue
    min_val = eligible_df[metric].min()
    max_val = eligible_df[metric].max()
    if max_val > min_val:
        eligible_df[f'normalized_{metric}'] = (eligible_df[metric] - min_val) / (max_val - min_val)
    else:
        eligible_df[f'normalized_{metric}'] = 0

# Calculate weighted score
eligible_df['score'] = 0
for metric, weight in weights.items():
    if metric not in metrics:
        continue
    eligible_df['score'] += eligible_df[f'normalized_{metric}'].fillna(0) * weight

# Rank projects
ranked_df = eligible_df.sort_values(by='score', ascending=False)

# Save results
output_path = os.path.join(directory_path, config['output']['rankings_file'])
ranked_df.to_csv(output_path, index=False)

print(f"Ranking complete! Check '{output_path}' for results.")