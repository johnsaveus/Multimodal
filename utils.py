from sklearn.model_selection import train_test_split
import os
import pandas as pd

def proccess_raw(raw_dir = 'Data/raw'):
    song_paths = []
    labels = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                label = os.path.basename(root)
                song_paths.append(full_path)
                labels.append(label)
    return song_paths, labels

def stratified_split(song_paths, labels, train_ratio = 0.8, val_ratio = 0.1, save_dir = 'Data/splits'):
    test_ratio = 1 - train_ratio - val_ratio
    songs_train, songs_val_test, y_train , y_val_test = train_test_split(song_paths,
                                                                     labels,
                                                                     test_size = val_ratio + test_ratio,
                                                                     random_state=42,
                                                                     stratify = labels)
    songs_val, songs_test, y_val , y_test = train_test_split(songs_val_test,
                                                             y_val_test,
                                                             test_size = 0.5 ,
                                                             random_state=42,
                                                             stratify = y_val_test)
    train_data = pd.DataFrame({'Path': songs_train, 'Label': y_train})
    val_data = pd.DataFrame({'Path': songs_val, 'Label': y_val})
    test_data = pd.DataFrame({'Path': songs_test, 'Label': y_test})

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_path = os.path.join(save_dir, 'train.csv')
    val_path = os.path.join(save_dir, 'val.csv')
    test_path = os.path.join(save_dir, 'test.csv')

    # Save the DataFrames as CSV files
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
### Create Splits

song_paths, labels = proccess_raw()
stratified_split(song_paths = song_paths, labels = labels)

