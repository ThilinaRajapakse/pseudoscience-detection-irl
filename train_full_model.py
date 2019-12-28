from simpletransformers.classification import ClassificationModel
import os
import pandas as pd
from sklearn.model_selection import train_test_split


irl_dir = 'irl_predictions'
other_dir = 'data_dfs_by_source'
irl_data_sources = os.listdir('irl_predictions')
other_sources = [source for source in os.listdir(other_dir) if source not in irl_data_sources]

irl_df = pd.concat([pd.read_csv(f'{irl_dir}/{source}/data_irl.tsv', sep='\t', index_col=[0]) for source in irl_data_sources])
other_df = pd.concat([pd.read_csv(f'{other_dir}/{source}/data_all.tsv', sep='\t', header=None) for source in other_sources])
other_df.columns = ['labels', 'text']


data_df = pd.concat([irl_df, other_df])
train_df, test_df = train_test_split(data_df, test_size=0.2)

train_df.to_csv('data/final_model/train.tsv', index=False, sep='\t')
test_df.to_csv('data/final_model/test.tsv', index=False, sep='\t')

train_args = {
    'max_seq_length': 512,
    'num_train_epochs': 5,
    'train_batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 5e-5,
    'save_steps': 50000,

    'sliding_window': True,
}

model = ClassificationModel('roberta', 'roberta-base', args=train_args)
model.train_model(train_df)