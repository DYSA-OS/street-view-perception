import requests
from tqdm import tqdm
from PIL import Image
import subprocess
import os
import pandas as pd
import shutil
import osmnx as ox
from sklearn.model_selection import train_test_split


# =======================================================================
# 1. Download DataSet
# =======================================================================

def download_dataset():
    url = 'https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1'
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, desc='Downloading archive', unit='iB', unit_scale=True)

    with open('place-pulse-2.0.zip', 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print('ERROR, something went wrong')

    subprocess.run('unzip place-pulse-2.0.zip -d place-pulse', shell=True)
    subprocess.run('rm -rf ./place-pulse/__MACOSX ', shell=True)
    subprocess.run('rm -rf place-pulse-2.0.zip', shell=True)

    print('[Done] - STEP 1: download dataset')

# =======================================================================
# 2. Extract DataSet
# =======================================================================
def extract_walk_road():
    df_qscores = pd.read_csv('./place-pulse/qscores.tsv', delimiter='\t')

    score_map = {
        '50a68a51fdc9f05596000002': 'safe',
        '50f62c41a84ea7c5fdd2e454': 'lively',
        '50f62c68a84ea7c5fdd2e456': 'clean',
        '50f62cb7a84ea7c5fdd2e458': 'wealthy',
        '50f62ccfa84ea7c5fdd2e459': 'depressing',
        '5217c351ad93a7d3e7b07a64': 'beautiful'
    }

    scores = df_qscores.pivot(index='location_id', columns='study_id', values='trueskill.stds.-1').reset_index()
    scores.columns = ['location_id'] + [score_map.get(col, 'beautiful') for col in scores.columns[1:]]

    filenames = os.listdir('./place-pulse/images')
    filename_data = [(f, f.split('_')[2].strip()) for f in filenames]

    df_files = pd.DataFrame(filename_data, columns=['filename', 'location_id'])
    df_merged = pd.merge(df_files, scores, on='location_id', how='left')

    df_merged['city'] = df_merged['filename'].apply(lambda x: x.split('_')[-1].split('.')[0])
    df_merged['lat'] = df_merged['filename'].apply(lambda x: x.split('_')[0])
    df_merged['long'] = df_merged['filename'].apply(lambda x: x.split('_')[1])

    locs = [
        'Amsterdam', 'Atlanta', 'Bangkok', 'Barcelona', 'Berlin', 'Boston',
        'Bratislava', 'Bucharest', 'Chicago', 'Denver', 'Dublin', 'Gaborone',
        'Guadalajara', 'Helsinki', 'HongKong', 'Houston', 'Kiev', 'Kyoto',
        'Lisbon', 'London', 'Madrid', 'Melbourne', 'Milan', 'Minneapolis',
        'Montreal', 'Moscow', 'Munich', 'Paris', 'Philadelphia', 'Portland',
        'Prague', 'RioDeJaneiro', 'Rome', 'Seattle', 'Singapore', 'Stockholm',
        'Sydney', 'Taipei', 'Toronto', 'Warsaw'
    ]

    df_city_list = []
    for loc in tqdm(locs):
        df_city = df_merged[df_merged['city'] == loc].copy()
        df_city['lat-long'] = df_city.apply(
            lambda col: f"{round(float(col['lat']), 4)}, {round(float(col['long']), 4)}", axis=1)

        try:
            G = ox.graph_from_place(loc, network_type='walk')
            df_walk = ox.graph_to_gdfs(G, edges=False).rename(columns={'y': 'lat', 'x': 'long'}).reset_index(drop=True)
            df_walk['lat-long'] = df_walk.apply(
                lambda col: f"{round(float(col['lat']), 4)}, {round(float(col['long']), 4)}", axis=1)

            merged_df = pd.merge(df_walk, df_city, on='lat-long', suffixes=('_highway', '_data'))
            merged_df['city'] = loc
            merged_df = merged_df[
                ['filename', 'city', 'lat_data', 'long_data', 'beautiful', 'safe', 'lively', 'clean', 'depressing',
                 'wealthy']].rename(columns={'lat_data': 'lat', 'long_data': 'long'})
            df_city_list.append(merged_df)

            subprocess.run('rm -rf ./cache', shell=True)
        except:
            pass

    df_dataset = pd.concat(df_city_list)

    source_folder = './place-pulse/images'
    destination_folder = './datasets'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    existing_files = set(os.listdir(source_folder))
    df_dataset['exists'] = df_dataset['filename'].isin(existing_files)
    filtered_df = df_dataset[df_dataset['exists']].drop(columns=['exists'])

    for filename in tqdm(filtered_df['filename'].unique(), total=filtered_df['filename'].nunique()):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy(source_path, destination_path)

    satisfaction_mapper = lambda val: 0 if val < 5 else (1 if val < 7 else 2)
    for col in ['beautiful', 'safe', 'lively', 'clean', 'depressing', 'wealthy']:
        filtered_df[col] = filtered_df[col].apply(satisfaction_mapper)

    filtered_df = filtered_df.drop_duplicates(subset='filename', keep='first')
    filtered_df.to_csv('data.csv', index=False)

    print('[Done] - STEP 2: extract walk road data')


# =======================================================================
# 3. Crop Google Logo
# =======================================================================
def crop_bottom(image_path, output_path, crop_height):
    with Image.open(image_path) as img:
        width, height = img.size
        box = (0, 0, width, height - crop_height)
        cropped_img = img.crop(box)
        cropped_img.save(output_path)


def process_images(input_dir, output_dir, crop_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.JPG')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            crop_bottom(input_path, output_path, crop_height)


# =======================================================================
# 4. Split Train, Val, Test Set
# =======================================================================
def split_dataset():
    df = pd.read_csv('data.csv')

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)

    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_ratio_temp, random_state=42)

    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_df.to_csv(os.path.join(data_dir, 'origin_train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'origin_val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'origin_test.csv'), index=False)

    print(f'Train set size: {len(train_df)}')
    print(f'Validation set size: {len(val_df)}')
    print(f'Test set size: {len(test_df)}')


def main():
    # STEP 1
    download_dataset()

    # STEP 2
    extract_walk_road()

    # STEP 3
    dir = './datasets'
    crop_height = 30
    process_images(dir, dir, crop_height)

    # STEP 4
    split_dataset()

    print('done')


if __name__ == '__main__':
    main()