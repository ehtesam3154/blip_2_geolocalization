import torch
from PIL import Image
import os
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
import json
import random

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


image_directory = "/lustre/fs1/home/cap6412.student24/datasets/WGV/val/"

with open("/lustre/fs1/home/cap6412.student24/datasets/WGV/labels_list.csv", 'r') as file:
    # for i in range(5):
    #     line = file.readline()
    #     print(line.strip())
    mapping = [x.strip().split(',') for x in file.readlines()[1:]]

cities = [" ".join(x[0].lower().split('_')) for x in mapping]
countries = [" ".join(x[2].split('_')) for x in mapping]
#print(list(zip(cities, countries)))

city_to_country = lambda x: countries[cities.index(x.lower())]

def get_options(country):
    options = [country]
    while len(options)<5:
        c = random.choice(countries)
        if c not in options:
            options.append(c)
    return options

raw_image_caption_pair = []

# this is for only random 100 images
image_files = os.listdir(image_directory)
random_image_files = random.sample(image_files, 100)
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
    # Construct the prompt string using join with "or" separator
    prompt = f"Question: Which one of these countries ({' or '.join(get_options(caption))}) is this image from? Answer:"
    
    #print(prompt)
    
    # Generate the answer using the model
    generated_answer = model.generate({'image': image, 'prompt': prompt})[0]
    
    # Append the generated answer and the caption to gen_answers list
    gen_answers.append([generated_answer, caption])

correct = 0

for x in gen_answers:
    print(x)
    if x[1].lower() in x[0].lower():
        correct += 1

accuracy = correct/len(gen_answers)

accuracy = "{:.4f}".format(accuracy)

with open("/home/cap6412.student24/LAVIS/task2_closedvqa.txt", 'a') as file:
	file.write(f"closed VQA Accuracy: {accuracy}")