import torch
from PIL import Image
import os
from lavis.models import load_model_and_preprocess
import json
import random
from tqdm import tqdm

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

with open("/lustre/fs1/home/cap6412.student24/datasets/WGV/labels_list.csv", 'r') as file:
    # for i in range(5):
    #     line = file.readline()
    #     print(line.strip())
    mapping = [x.strip().split(',') for x in file.readlines()[1:]]

cities = [" ".join(x[0].lower().split('_')) for x in mapping]
countries = [" ".join(x[2].split('_')) for x in mapping]
#print(list(zip(cities, countries)))

city_to_country = lambda x: countries[cities.index(x.lower())]

image_directory = "/lustre/fs1/home/cap6412.student24/datasets/WGV/val/"

raw_image_caption_pair = []

#for fname in os.listdir(image_directory):
#    image = Image.open(os.path.join(image_directory, fname)).convert('RGB')
#    country = city_to_country(' '.join(fname.split('_')[:-2]))
#    raw_image_caption_pair.append([image, country])

# this is for only random 100 images
image_files = os.listdir(image_directory)
random_image_files = random.sample(image_files, 10)
for fname in random_image_files:
    image = Image.open(os.path.join(image_directory, fname)).convert('RGB')
    country = city_to_country(' '.join(fname.split('_')[:-2]))
    raw_image_caption_pair.append([image, country])

#load pretrained blip_2 model
model, vis_processors, _ = load_model_and_preprocess(name = 'blip2_opt', model_type='pretrain_opt2.7b', is_eval = True, device = device)
# print('Model loaded succesfully')

#prepare images and captions
# images_and_captions = [[vis_processors['eval'](raw_image).unsqueeze(0).to(device), caption] for raw_image, caption in raw_image_caption_pair]
images_and_captions = []
for raw_image, caption in tqdm(raw_image_caption_pair, desc = 'Processing Images', unit = 'image'):
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    images_and_captions.append([image, caption])

gen_answers = []

for image, caption in tqdm(images_and_captions, desc = 'generating answers', unit = 'image'):
    #generated_answer = model.generate({"image": image, "prompt": 'Question: Which country is this image from? Full name please. Answer:'})[0]
    generated_answer = model.generate({"image": image, "prompt": 'Question: Which country do you think this image be from? Answer:'})[0]
    #print(generated_answer)
    gen_answers.append([generated_answer, caption])

correct = 0
for x in tqdm(gen_answers, desc = 'processing answers', unit = 'answers'):
    print(x)
    if x[1].lower() in x[0].lower():
        correct += 1

accuracy = correct/len(gen_answers)

with open("/home/cap6412.student24/LAVIS/task2_openvqa.txt", 'a') as file:
	file.write(f"Open VQA Accuracy: {accuracy}")