from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch
import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from tqdm import tqdm
import pandas as pd
import re

model_path = '4bit/llava-v1.5-13b-3GB'
kwargs = {'device_map': 'auto'}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor


def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output


def alpha_num(title):
    return re.sub(r'[^A-Za-z0-9 ]', '', title)


stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
             "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
             "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
             "has",
             "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
             "it", "it's", "its",
             "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
             "should", "so", "some",
             "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
             "there's", "these", "they",
             "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
             "until", "up", "very", "was", "we",
             "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who",
             "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves",
             "pedestrians", "perspective", "image", "depicts", "street", "view", "pedestrian", "describe",
             "pictures", "picture"
             ]


def remove_stopwords(title):
    final_text = []
    for i in title.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return ' '.join(final_text)


def delete_duplications(text):
    words = text.split()

    unique_words = []

    for word in words:
        if word not in unique_words:
            unique_words.append(word)

    result = ' '.join(unique_words)
    return result


def main():
    data_dir = '../data'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    for data in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(data_dir, f'origin_{data}.csv'))

        prompts = []
        for file_name in tqdm(df['filename']):
            try:
                image, output = caption_image(f'../datasets/{file_name}',
                                              'Describe this street view image from a pedestrians perspective. Describe the weather, surrounding natural features, road conditions, presence of pedestrians or vehicles, type and condition of buildings, cleanliness of the street, vegetation such as street trees, and overall ambiance of the image.')
                prompt = output.strip().replace('.\n', ' ')
                prompts.append(prompt)

            except:
                prompts.append('')
                continue

        df['prompt'] = prompts
        df['prompt'] = df['prompt'].str.lower()
        df['prompt'] = df['prompt'].apply(alpha_num).apply(remove_stopwords).apply(delete_duplications)

        df.to_csv('prompt.csv', index=False)

        df.to_csv(os.path.join(data_dir, f'prompt_{data}.csv'), index=False)


if __name__ == '__main__':
    main()