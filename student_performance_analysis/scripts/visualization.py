import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('../data/StudentsPerformance.csv')

# 1. Score Distribution Histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='math score', kde=True, color='blue', label='Math')
sns.histplot(data=df, x='reading score', kde=True, color='green', label='Reading')
sns.histplot(data=df, x='writing score', kde=True, color='red', label='Writing')
plt.title('Distribution of Test Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('../visualizations/score_distribution.png')
plt.close()

# 2. Gender Performance Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='value', hue='variable',
            data=pd.melt(df, id_vars=['gender'], value_vars=['math score', 'reading score', 'writing score']))
plt.title('Performance by Gender')
plt.ylabel('Score')
plt.savefig('../visualizations/gender_performance.png')
plt.close()

# 3. Parental Education Impact Stacked Bar Chart
education_order = ['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree']
df['parental level of education'] = pd.Categorical(df['parental level of education'], categories=education_order, ordered=True)
avg_scores = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()

plt.figure(figsize=(12, 6))
avg_scores.plot(kind='bar', stacked=True)
plt.title('Average Scores by Parental Education Level')
plt.xlabel('Parental Education Level')
plt.ylabel('Average Score')
plt.legend(title='Subject')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../visualizations/parental_education_impact.png')
plt.close()

# 4. Test Preparation Effect Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='math score', y='reading score', hue='test preparation course', style='test preparation course')
plt.title('Math vs Reading Scores: Impact of Test Preparation')
plt.savefig('../visualizations/test_prep_effect.png')
plt.close()

# 5. Race/Ethnicity Performance Radar Chart
avg_scores_by_race = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean()

angles = np.linspace(0, 2*np.pi, len(avg_scores_by_race.columns), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for race in avg_scores_by_race.index:
    values = avg_scores_by_race.loc[race].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=race)
    ax.fill(angles, values, alpha=0.25)

ax.set_thetagrids(angles[:-1] * 180/np.pi, avg_scores_by_race.columns)
ax.set_ylim(0, 100)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.title('Average Scores by Race/Ethnicity')
plt.tight_layout()
plt.savefig('../visualizations/race_ethnicity_comparison.png')
plt.close()