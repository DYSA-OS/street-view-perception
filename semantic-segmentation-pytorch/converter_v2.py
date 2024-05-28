import os
import csv
import torch
import numpy as np
from scipy import io
from PIL import Image
from torchvision import transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from tqdm import tqdm
import pandas as pd


colors = io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def draw_mask(img, pred, save_path, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    im_vis_image = Image.fromarray(im_vis)
    im_vis_image.save(save_path)


def count_pixcel(img, pred, index):
    counts = 0
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1

    total_pixels = img.shape[0] * img.shape[1]
    class_pixel_count = np.sum(pred == index)
    class_ratio = class_pixel_count / total_pixels * 100
    counts += class_ratio

    return names[index+1], counts

# ========================================================================
# Loading the segmentation model
# ========================================================================

# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

# ========================================================================
# Loading test data
# ========================================================================
dataset_dir = '../datasets'
for filename in tqdm(os.listdir(dataset_dir)):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.endswith(('.JPG')):
        image_path = os.path.join(dataset_dir, filename)

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    pil_image = Image.open(image_path).convert('RGB')
    img_original = np.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # ========================================================================
    # Running model
    # ========================================================================
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    # # Masking entire classes
    # # ========================================================================
    # draw_mask(img_original, pred, os.path.join(output_dir, 'test.png'))
    #
    # # Masking specific class
    # # ========================================================================
    # predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
    #
    # top_1_class = predicted_classes[0]
    # draw_mask(img_original, pred, os.path.join(output_dir, 'test_top_1.png'), top_1_class)
    # count_pixcel(img_original, pred, top_1_class)

    # Making classes pixel ratio
    predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]

    result_list = [f'{filename}']
    for c, pred_class in enumerate(predicted_classes[:10]):

        class_name, class_ratio = count_pixcel(img_original, pred, pred_class) # return: class_name, pixel_ratio

        result_list.append(f'{class_name} {class_ratio:.5f}')

        log_entry = ' '.join(result_list)
        log_entry += '\n'


    # test.png building 44.28303 car 17.40813 sky 16.93228 tree 15.27506.....
    with open('convert_v2.txt', 'a') as f:
        f.write(log_entry)


def make_dataset():
    file_path = './convert_v2.txt'
    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_path = parts[0]
            classes = parts[1::2]
            values = parts[2::2]

            row_data = {'image_path': image_path}
            for cls, value in zip(classes, values):
                row_data[cls] = float(value)

            data.append(row_data)

    df = pd.DataFrame(data)
    df = df.fillna(0).rename(columns={'image_path': 'filename'})

    for class_name in ['beautiful', 'clean']:
        for data in ['train', 'val', 'test']:
            df2 = pd.read_csv(f'../balanced_data/origin_{data}_{class_name}.csv')

            merged_df = pd.merge(df, df2, on=['filename'], how='left').dropna()
            merged_df.to_csv(f'../balanced_data/hrnet_{data}_{class_name}.csv', index=False)


def main():
    make_dataset()


if __name__ == '__main__':
    main()