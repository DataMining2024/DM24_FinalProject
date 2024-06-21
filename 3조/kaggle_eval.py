import argparse
import torch
import os
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from sklearn.metrics import accuracy_score

import numpy as np

from random import choice
import cv2
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import csv
import time

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def cv2_jpg(img, quality):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def png2jpg(img, quality):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': png2jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    asdf = os.path.join(args.base_path,'submission_' + args.target_label + '_' + str(args.top_n) + '_' + str(time.strftime('%Y.%m.%d')))

    f = open(args.csv_path, 'r')
    f2 = open(asdf+'.csv', 'w', newline='')

    rdr = csv.reader(f)
    wr = csv.writer(f2)
    feature_names = None
    label_names = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']

    assert args.target_label in label_names, "feature_name should be one of ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']"

    wr.writerow(['id', args.target_label])

    for idx, line in tqdm(enumerate(rdr)):
        print('current is {}'.format(idx))
        if idx == 0:
            feature_names = line[2:]
        else:
            features = ""

            for feature_idx, feature in enumerate(feature_names):
                features += "F{}: {} \n ".format(feature_idx, line[feature_idx + 2])

            image_file = os.path.join(args.img_path,"{}.jpeg".format(line[1]))
            image = Image.open(image_file).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]

            current_result = []
            current_result.append(line[1])

            qs = "Predict {} \n {} ".format(args.target_label, features)

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            current_result.append(outputs)

            wr.writerow(current_result)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/LLaVA/scripts/v1_5/checkpoints/llava-v1.5-7b-kaggle-x4-top10-lora")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--base-path", type=str, default="/LLaVA/kaggle/submission")
    parser.add_argument("--exp-name", type=str, default="debug")

    parser.add_argument("--csv_path", type=str, default='/home/kaggle/test/X4_mean_10.csv')
    parser.add_argument("--img_path", type=str, default='/home/kaggle/test/test_images')
    parser.add_argument("--target_label", type=str, default='X4')
    parser.add_argument("--top_n", type=int, default=10)

    args = parser.parse_args()

    eval_model(args)
