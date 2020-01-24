import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def read_files(input_dir, label, start_line=1, cleaner=None):
    data = []
    with os.scandir(input_dir) as files:
        for file in files:
            if file.name.endswith('.txt'):
                with open(file, encoding='utf-8') as f:
                    lines = [line.replace('\n', '')
                             for line in f if len(line.replace('\n', ''))]
                lines = lines[start_line:]
                text = ' '.join(lines)

                if cleaner:
                    text = cleaner(text)

                if len(text) > 100:
                    data.append([text, label])
                    
    return data


def clean_reuters(text):
    return '-'.join(text.split(' - ')[1:])


# Dictionary items are label and start_line respectively
scraped_dict = {
    'popsci': (0, 0),
    'sciencedaily': (0, 1),
    'smithsonian': (0, 0),
    'gizmodo': (0, 1),
    'mercola': (1, 1),
    'naturalnews': (1, 1),
    'collectiveevolution': (1, 1),
    'davidwolfe': (1, 1),
    'goop': (1, 1),
    'danachild': (1, 1),
    'foodbabe': (1, 1),
    'greenmedinfo': (1, 1),
    'thinkingmomsrevolution': (1, 1),
    'infowars': (1, 1),
    'greggbraden': (1, 1),
    'sciencebasedmedicine': (0, 1),
    'factcheck': (0, 1),    
    'snopes': (0, 1),    
    'reuters': (0, 1),
    'pnas': (0, 1)
}

# def form_dataset(sources):
#     data = []
#     for input_dir, info in scraped_dict.items():
#         if input_dir not in sources:
#             continue
#         label, start_line = info
#         input_dir = f'data/{input_dir}/scraped'
#         data.append(read_files(input_dir, label, start_line))
        
#     return data


datasets = {}
for input_dir, info in tqdm(scraped_dict.items()):
    label, start_line = info
    name = input_dir

    if name == 'reuters':
        cleaner = clean_reuters
    else:
        cleaner = None

    input_dir = f'data/{input_dir}/scraped'
    datasets[f'{name}'] = read_files(input_dir, label, start_line, cleaner)


prefix = 'data_dfs_by_source/'

for name, data in tqdm(datasets.items()):
    if not os.path.exists(prefix + name):
        os.makedirs(prefix + name)
    df = pd.DataFrame(data, columns=['text', 'labels'])

    df_bert = pd.DataFrame({
                            'labels': df['labels'],
                            'text': df['text'].replace(r'\n', ' ', regex=True)})

    df_bert = df_bert.drop_duplicates(subset='text')

    df_bert.to_csv(prefix + f'{name}/data_all.tsv', sep='\t', index=False)

    train, dev = train_test_split(df_bert, test_size=0.2)

    train = pd.DataFrame({
        'labels':train['labels'],
        'text': train['text'].replace(r'\t', ' ', regex=True)
    })

    train.to_csv(prefix + f'{name}/train.tsv', sep='\t', index=False)

    dev = pd.DataFrame({
        'labels':dev['labels'],
        'text': dev['text'].replace(r'\t', ' ', regex=True)
    })

    dev.to_csv(prefix + f'{name}/dev.tsv', sep='\t', index=False)