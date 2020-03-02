from simpletransformers.classification import ClassificationModel
import os
import pandas as pd
import datetime

try:
    completed = os.listdir('irl_predictions')
except:
    os.makedirs('irl_predictions')
    completed = []

train_args = {
    'max_seq_length': 512,
    'num_train_epochs': 1,
    'train_batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 4e-5,
    'save_steps': 50000,

    'sliding_window': True,
    'reprocess_input_data': True,
}


for irl_model in os.listdir('irl_models'):
    if irl_model not in completed:
        target_df = pd.read_csv(f'data_dfs_by_source/{irl_model}/data_all.tsv', sep='\t')
        target_df.columns = ['labels', 'text']
        model = ClassificationModel('roberta', f'irl_models/{irl_model}', args=train_args)
        to_predict = target_df['text'].tolist()
        predictions, _ = model.predict(to_predict)
        target_df['labels'] = predictions

        os.makedirs(f'irl_predictions/{irl_model}')
        target_df.to_csv(f'irl_predictions/{irl_model}/data_irl.tsv', sep='\t', index=False)
        exit()


with open("done.predictions", 'w') as f:
        f.write(f"Done at {datetime.datetime.now()}")
