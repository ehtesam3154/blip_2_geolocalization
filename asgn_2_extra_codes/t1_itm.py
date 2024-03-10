from tqdm import tqdm
import torch
from PIL import Image
import os
from lavis.models import load_model_and_preprocess
import json
import random
#import sacrebleu

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

with open("/lustre/fs1/home/cap6412.student24/datasets/WGV/labels_list.csv", 'r') as file:
    # for i in range(5):
    #     line = file.readline()
    #     print(line.strip())
    mapping = [x.strip().split(',') for x in file.readlines()[1:]]

print(f'Length of mapping: {len(mapping)}')

image_directory = "/lustre/fs1/home/cap6412.student24/datasets/WGV/val/"

cities = [' '.join(x[0].lower().split('_')) for x in mapping]
countries = [' '.join(x[2].lower().split('_')) for x in mapping]
city_to_country = lambda x: countries[cities.index(x.lower())]

templates = ["I took this photo in {}", 
             "{} is one of my favorite travel destinations",
               "A photo from my home country of {}",
                 "My wife sent me this photo from {}",
                   "I went to {} with my wife",
                     "captioned: {}"]


raw_image_caption_pair = []

#for fname in os.listdir(image_directory):
#    image = Image.open(os.path.join(image_directory, fname)).convert('RGB')
#    country = city_to_country(' '.join(fname.split('_')[:-2]))
#    raw_image_caption_pair.append([image, country])
    

# this is for only random 100 images
image_files = os.listdir(image_directory)
random_image_files = random.sample(image_files, 10)
for fname in random_image_files:
    #print(fname)
    image = Image.open(os.path.join(image_directory, fname)).convert('RGB')
    country = city_to_country(' '.join(fname.split('_')[:-2]))
    raw_image_caption_pair.append([image, country])
    
#load pretrained blip_2 model
model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_image_text_matching", model_type="pretrain", is_eval=True, device=device)
print('Model loaded succesfully')

countries = list(set(countries))

#utilize tempolates to create captions
raw_texts_and_captions = [[template.format(country), country] for template in templates for country in countries]

#prepare images and captions
images_and_captions = []
for raw_image, caption in tqdm(raw_image_caption_pair, desc = 'Processing Images', unit = 'image'):
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
#    print(f"Processed image shape: {image.shape}")
#    print(f"Caption: {caption}") 
    images_and_captions.append([image, caption])

#print(images_and_captions)

#prepare texts
texts_and_captions = []
for raw_text, caption in tqdm(raw_texts_and_captions, desc= 'Processing texts', unit = 'text'):
    text = text_processors['eval'](raw_text)
#    print(f"Processed text: {text}")
#    print(f"Caption: {caption}")
    texts_and_captions.append([text, caption])

 
def match_score(img, txt):
    #pass image and text through the model to get the raw output
    itm_output = model({'image': img, 'text_input': txt}, match_head = 'itm')
    score = itm_output[0][0].item()
    #apply softmax
    #itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    #print(itm_scores)

    #select probability corresponding to the positive class
    #score = itm_scores[:, 1].item()

    # print(f'The image and text are matched with a probability of {positive_probability:.3%}')

    return score

n = 1 #Only the top-1 is to be considered.

correct = 0

#for image, caption in tqdm(images_and_captions, desc="Matching Captions", unit="image"):
#    match_found = False
#    for candidate, caption in sorted(texts_and_captions, key = lambda x: match_score(image, x[0]))[:n]:
#        print(candidate, caption)
#        if caption == candidate:
#            match_found = True
#
#        if match_found:
#            correct += 1

for image, caption in images_and_captions:
      top_preds = sorted(texts_and_captions, key = lambda x: match_score(image, x[0]))[:n]
      #print(top_preds)
      for pred in top_preds:
            print(pred, caption)
            #print(type(caption), type(pred[1]))
            if pred[1] == caption:
                  #print('Found a match!')
                  correct += 1
                  break

#def sort_by_match_score(x):
#    return match_score(image, x[0])
#
#for image, caption in tqdm(images_and_captions, desc="Matching Captions", unit="image"):
#    top_preds = sorted(texts_and_captions, key=sort_by_match_score)[:n]
#    for pred in top_preds:
#        max_score = float('-inf')
#        max_caption = None
#        for x in texts_and_captions:
#            score = match_score(image, x[0])
#            if score > max_score:
#                max_score = score
#                max_caption = x[1]
#        if max_caption == caption:
#            correct += 1
#            break


#for image, caption in tqdm(images_and_captions, desc="Matching Captions", unit="image"):
#    top_preds = []  # Create an empty list to store top predictions
#
#    for text, pred_caption in texts_and_captions:  # Iterate through each text-caption pair
#        score = match_score(image, text)  # Calculate match score for the current pair
#        top_preds.append((score, text, pred_caption))  # Add score, text, and caption to the list
#
#    top_preds.sort(reverse=True)  # Sort in descending order of scores
#    top_preds = top_preds[:n]  # Take only the top n predictions
#
#    for _, _, pred_caption in top_preds:  # Iterate through the top predictions
#        if pred_caption == caption:
#            correct += 1
#            break


accuracy = correct/len(images_and_captions)
#accuracy = correct/100

with open("/home/cap6412.student24/LAVIS/task1_itm.txt", 'a') as file:
	file.write(f"Accuracy (Retrieval@{n}): {accuracy}\n")
