from simpletransformers.classification import ClassificationModel
import os
import pandas as pd
from tqdm.auto import tqdm
import json
import datetime


def run_trainers(bucket_dir, train_args=None):

    os.makedirs('irl_models', exist_ok=True)

    if os.path.isfile('completed_irl.txt'):
        with open("completed_irl.txt", 'r') as f:
            done = [d.replace('\n','') for d in f.readlines()]
    else:
        open('completed_irl.txt', 'a').close()
        with open("completed_irl.txt", 'r') as f:
            done = [d.replace('\n','') for d in f.readlines()]
    for train_file in os.listdir(bucket_dir):
        print(train_file[5:])
        print(done)
        if train_file[5:] not in done:
            train_df = pd.read_csv(bucket_dir + '/' + train_file + '/data_all.tsv', sep='\t')
            train_args['output_dir'] = f'irl_models/{train_file[5:]}/'
            train_args['cache_dir'] = f'cache_{train_file[5:]}/'

            train_args.update({'wandb_kwargs': {'name': train_file[5:]}})

            model = ClassificationModel('roberta', 'roberta-base', args=train_args)
            print(train_df.head())
            model.train_model(train_df)

            with open("completed_irl.txt", 'a') as f:
                f.write(f"{train_file[5:]}\n")
            exit()

    with open("done.runs", 'w') as f:
        f.write(f"Done at {datetime.datetime.now()}")


train_args = {
    'max_seq_length': 512,
    'num_train_epochs': 1,
    'train_batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 5e-5,
    'save_steps': 50000,
    
    'reprocess_input_data': True,
    'overwrite_output_dir': True,

    'sliding_window': True,

    'wandb_project': 'pseudo-with-irl-sliding-window',
}

run_trainers('data_dfs_for_irl', train_args=train_args)