#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


SAVE_FIGURES = False
RESULTS_DIR = './results_paper'


# In[3]:


EXPERIMENT = 'bivariate_power'
TAG = '_paper'
df1 = pd.read_csv(f'{RESULTS_DIR}/{EXPERIMENT}_results{TAG}.csv', sep=', ', engine='python')
df1['Experiment'] = 'Additive'

EXPERIMENT = 'bivariate_multiplic_power'
TAG = ''
df2 = pd.read_csv(f'{RESULTS_DIR}/{EXPERIMENT}_results{TAG}.csv', sep=', ', engine='python')
df1['Experiment'] = 'Multiplic.'

df = pd.concat([df1, df2])


# In[4]:


plot_df = df

x_var_rename_dict = {
    'sample_size': '# Samples',
    'Number of environments': '# Environments',
    'Fraction of shifting mechanisms': 'Shift fraction',
    'dag_density': 'Edge density',
    'n_variables': '# Variables',
}

plot_df = df.rename(
        x_var_rename_dict, axis=1
    ).rename(
        {'Method': 'Test', 'Soft': 'Score'}, axis=1
    ).replace(
        {
            'er': 'Erdos-Renyi',
            'ba': 'Hub',
            'PC (pool all)': 'Full PC (oracle)',
            'Full PC (KCI)': r'Pooled PC (KCI) [25]',
            'Min changes (oracle)': 'MSS (oracle)',
            'Min changes (KCI)': 'MSS (KCI)',
            'Min changes (GAM)': 'MSS (GAM)',
            'Min changes (Linear)': 'MSS (Linear)',
            'Min changes (FisherZ)': 'MSS (FisherZ)',
            'MC': r'MC [11]',
            False: 'Hard',
            True: 'Soft',
        }
)

plot_df = plot_df.loc[
    (~plot_df['Test'].isin(['Full PC (oracle)', 'MSS (oracle)'])) &
    (plot_df['# Environments'] == 2) &
    (plot_df['Score'] == 'Hard')
]

plot_df = plot_df.replace({
    '[[];[0]]': 'P(X1)',
    '[[];[1]]': 'P(X2|X1)',
    '[[];[]]': 'Neither',
    '[[];[0;1]]': 'Both',
})


# In[5]:


sns.set_context('paper')
fig, axes = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(7.5, 2.5))

intv_targets = ['P(X1)', 'P(X2|X1)', 'Neither', 'Both']
ax_var = 'intervention_targets'
x_var = 'Precision' # 'False orientation rate' # 
y_var = 'Recall' # 'True orientation rate'# 
hue = 'Test'

for targets, ax in zip(intv_targets, axes.flatten()):
    mean_df = plot_df[plot_df[ax_var] == targets].groupby('Test').mean().reset_index()
    std_df = plot_df[plot_df[ax_var] == targets].groupby('Test')[['Precision', 'Recall']].std().reset_index()
    std_df.rename(
        {'Precision': 'Precision std', 'Recall': 'Recall std'}, axis=1
    )
    
    g = sns.scatterplot(
        data=plot_df[plot_df[ax_var] == targets].groupby('Test').mean().reset_index(),
        x=x_var,
        y=y_var,
        hue=hue,
        ax=ax,
        palette=[
            sns.color_palette("tab10")[i]
            for i in [2, 3, 4, 5, 7, 6] # 3, 4, 5, 
        ],
        hue_order=[
            'MSS (KCI)',
            'MSS (GAM)',
            'MSS (FisherZ)',
            'MSS (Linear)',
            'Pooled PC (KCI) [25]',
            'MC [11]',
        ],
        legend='full',
        s=100
    )
    ax.set_title(f'Shift in {targets}')
    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
for ax in axes[:-1]:
    ax.get_legend().remove()
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig('./figures/bivariate_power_plots.pdf')
plt.show()


# ## multiplicative

# In[7]:


EXPERIMENT = 'bivariate_multiplic_power'
TAG = ''
df = pd.read_csv(f'{RESULTS_DIR}/{EXPERIMENT}_results{TAG}.csv', sep=', ', engine='python')


# In[8]:


plot_df = df

x_var_rename_dict = {
    'sample_size': '# Samples',
    'Number of environments': '# Environments',
    'Fraction of shifting mechanisms': 'Shift fraction',
    'dag_density': 'Edge density',
    'n_variables': '# Variables',
}

plot_df = df.rename(
        x_var_rename_dict, axis=1
    ).rename(
        {'Method': 'Test', 'Soft': 'Score'}, axis=1
    ).replace(
        {
            'er': 'Erdos-Renyi',
            'ba': 'Hub',
            'PC (pool all)': 'Full PC (oracle)',
            'Full PC (KCI)': r'Pooled PC (KCI) [25]',
            'Min changes (oracle)': 'MSS (oracle)',
            'Min changes (KCI)': 'MSS (KCI)',
            'Min changes (GAM)': 'MSS (GAM)',
            'Min changes (Linear)': 'MSS (Linear)',
            'Min changes (FisherZ)': 'MSS (FisherZ)',
            'MC': r'MC [11]',
            False: 'Hard',
            True: 'Soft',
        }
)

plot_df = plot_df.loc[
    (~plot_df['Test'].isin(['Full PC (oracle)', 'MSS (oracle)'])) &
    (plot_df['# Environments'] == 2) &
    (plot_df['Score'] == 'Hard')
]

plot_df = plot_df.replace({
    '[[];[0]]': 'P(X1)',
    '[[];[1]]': 'P(X2|X1)',
    '[[];[]]': 'Neither',
    '[[];[0;1]]': 'Both',
})


# In[9]:


sns.set_context('paper')
fig, axes = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(7.5, 2.5))

intv_targets = ['P(X1)', 'P(X2|X1)', 'Neither', 'Both']
ax_var = 'intervention_targets'
x_var = 'Precision' # 'False orientation rate' # 
y_var = 'Recall' # 'True orientation rate'# 
hue = 'Test'

for targets, ax in zip(intv_targets, axes.flatten()):
    mean_df = plot_df[plot_df[ax_var] == targets].groupby('Test').mean().reset_index()
    std_df = plot_df[plot_df[ax_var] == targets].groupby('Test')[['Precision', 'Recall']].std().reset_index()
    std_df.rename(
        {'Precision': 'Precision std', 'Recall': 'Recall std'}, axis=1
    )
    
    g = sns.scatterplot(
        data=plot_df[plot_df[ax_var] == targets].groupby('Test').mean().reset_index(),
        x=x_var,
        y=y_var,
        hue=hue,
        ax=ax,
        palette=[
            sns.color_palette("tab10")[i]
            for i in [2, 3, 4, 5, 7, 6] # 3, 4, 5, 
        ],
        hue_order=[
            'MSS (KCI)',
            'MSS (GAM)',
            'MSS (FisherZ)',
            'MSS (Linear)',
            'Pooled PC (KCI) [25]',
            'MC [11]',
        ],
        legend='full',
        s=100
    )
    ax.set_title(f'Shift in {targets}')
    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
for ax in axes[:-1]:
    ax.get_legend().remove()
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig('./figures/bivariate_multiplic_power_plots.pdf')
plt.show()

