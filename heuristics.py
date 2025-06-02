import pandas as pd
import yaml
import json
import os

# Define directory path
directory_path = r'C:\Users\USER\Desktop\Onchain Labs\Retro Funding'

# Load YAML configuration
yaml_path = os.path.join(directory_path, 'heuristics.yaml')
with open(yaml_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Extract weights, thresholds, and file names
weights = config['model']['parameters']['weights']
thresholds = config['model']['parameters']['thresholds']
projects_file = config['data']['projects_file']
contributor_file = config['data']['contributor_file']
dependency_file = config['data']['dependency_file']

# Construct file paths
projects_path = os.path.join(directory_path, projects_file)
contributor_path = os.path.join(directory_path, contributor_file)
dependency_path = os.path.join(directory_path, dependency_file)

# Load data
projects_df = pd.read_csv(projects_path)

# Load JSON for contributors
with open(contributor_path, 'r', encoding='utf-8') as file:
    contributor_data = json.load(file)
contributor_df = pd.DataFrame(contributor_data)

#dependents_df = pd.read_csv(onchain_metrics_path)
dependents_df = pd.read_csv(dependency_path)

# Compute dependents: count occurrences of each project_id
dependents_count = dependents_df.groupby('project_id').size().reset_index(name='dependents')

# Merge data
merged_df = projects_df[['project_id', 'star_count', 'fork_count']]

# Merge contributors using developer_connection_count
merged_df = merged_df.merge(
    contributor_df[['project_id', 'developer_connection_count']],
    on='project_id',
    how='left'
)

# Merge dependents (project_id matches project_id)
merged_df = merged_df.merge(
    dependents_count[['project_id', 'dependents']],
    left_on='project_id',
    right_on='project_id',
    how='left'
)

# Rename columns
merged_df = merged_df.rename(columns={
    'star_count': 'stars',
    'fork_count': 'forks',
    'developer_connection_count': 'contributors',
    'dependents': 'dependents'
})

# Filter projects based on thresholds (min_dependents: 1, min_contributors: 1)
eligible_df = merged_df[
    (merged_df['dependents'].fillna(0) >= thresholds['min_dependents']) &
    (merged_df['contributors'].fillna(0) >= thresholds['min_contributors'])
]

# Normalize metrics to 0-1 scale
metrics = ['stars', 'forks', 'contributors', 'dependents']
for metric in metrics:
    if metric in eligible_df.columns:
        min_val = eligible_df[metric].min()
        max_val = eligible_df[metric].max()
        if max_val > min_val:
            eligible_df[f'normalized_{metric}'] = (eligible_df[metric] - min_val) / (max_val - min_val)
        else:
            eligible_df[f'normalized_{metric}'] = 0
    else:
        eligible_df[f'normalized_{metric}'] = 0

# Calculate weighted score
eligible_df['score'] = 0
for metric, weight in weights.items():
    if metric in metrics:
        eligible_df['score'] += eligible_df[f'normalized_{metric}'].fillna(0) * weight

# Rank projects by score
ranked_df = eligible_df.sort_values(by='score', ascending=False)

# Save results
output_path = os.path.join(directory_path, config['output']['rankings_file'])
ranked_df.to_csv(output_path, index=False)

print(f"Ranking complete! Check '{output_path}' for results.")