from simpletransformers.classification import ClassificationModel
import pandas as pd


eval_df = pd.read_csv('data/final_model/test.tsv', sep='\t')
print(eval_df.head())

train_args = {
    'max_seq_length': 512,
    'num_train_epochs': 5,
    'train_batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 5e-5,
    'save_steps': 50000,

    'sliding_window': True,
}

model = ClassificationModel('roberta', 'outputs', args=train_args)
model.eval_model(eval_df, verbose=True)
