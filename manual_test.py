import os
from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import f1_score


texts = []
labels = []
with os.scandir('manual') as files:
    input_files = files
    for i, file in enumerate(files):
        if file.name.endswith('.txt'):
            with open(file, encoding='utf-8') as f:
                lines = [line.replace('\n', '')
                         for line in f if len(line.replace('\n', ''))]
            try:
                label = int(lines[0])
                lines = lines[1:]
            except:
                label = int(lines[1])
                lines = lines[2:]
            text = ' '.join(lines)
            texts.append(text)
            labels.append(label)

df = pd.DataFrame({'labels': labels, 'text': texts})
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
model.eval_model(df, verbose=True, f1_score=f1_score)

