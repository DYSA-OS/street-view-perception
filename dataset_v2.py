import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os


def make_origin_balanced():
    result_dir = './balanced_data'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    df = pd.read_csv('./data.csv')
    print(f'origin df len: {len(df)}')

    for class_name in ['beautiful', 'clean']:
        data = df[['filename', class_name]]

        if len(data[class_name].unique()) == 3:
            class_0 = data[data[class_name] == 0]
            class_1 = data[data[class_name] == 1]
            class_2 = data[data[class_name] == 2]

            # 클래스 0, 1, 2의 샘플 수 중 가장 작은 값을 기준으로 언더샘플링
            min_class_size = min(len(class_0), len(class_1), len(class_2))

            class_0_downsampled = resample(class_0, replace=False, n_samples=min_class_size, random_state=42)
            class_1_downsampled = resample(class_1, replace=False, n_samples=min_class_size, random_state=42)
            class_2_downsampled = resample(class_2, replace=False, n_samples=min_class_size, random_state=42)

            balanced_data = pd.concat([class_0_downsampled, class_1_downsampled, class_2_downsampled])
            balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            class_0 = data[data[class_name] == 0]
            class_1 = data[data[class_name] == 1]

            # 클래스 0, 1, 2의 샘플 수 중 가장 작은 값을 기준으로 언더샘플링
            min_class_size = min(len(class_0), len(class_1))

            class_0_downsampled = resample(class_0, replace=False, n_samples=min_class_size, random_state=42)
            class_1_downsampled = resample(class_1, replace=False, n_samples=min_class_size, random_state=42)

            balanced_data = pd.concat([class_0_downsampled, class_1_downsampled])
            balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f'converted {class_name} df len: {len(balanced_data)}')

        train_ratio = 0.7
        val_ratio = 0.1
        test_ratio = 0.2

        train_df, temp_df = train_test_split(balanced_data, test_size=1 - train_ratio, random_state=42, stratify=balanced_data[class_name])

        val_ratio_temp = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=1 - val_ratio_temp, random_state=42, stratify=temp_df[class_name])

        train_df.to_csv(os.path.join(result_dir, f'origin_train_{class_name}.csv'), index=False)
        val_df.to_csv(os.path.join(result_dir, f'origin_val_{class_name}.csv'), index=False)
        test_df.to_csv(os.path.join(result_dir, f'origin_test_{class_name}.csv'), index=False)

        print(f'Train set size: {len(train_df)}')
        print(f'Validation set size: {len(val_df)}')
        print(f'Test set size: {len(test_df)}')

make_origin_balanced()