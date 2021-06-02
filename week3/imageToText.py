import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'#your path to tesseract for windows users
from PIL import Image
import operator
import cv2 as cv
import jellyfish as jel
import os

from string import ascii_letters, digits

def get_bbdd_texts(bbdd_path):
    bbdd_texts = {}
    for image_filename in sorted(os.listdir(bbdd_path)):
        if image_filename.endswith('.txt'):
            image_id = int(image_filename.replace('.txt', '').replace('bbdd_', ''))
            f = open(os.path.join(bbdd_path, image_filename), "r")
            line= f.readline()
            if line.strip():
                bbdd_text = line.lower().replace("(","").replace("'"," ").replace(")","") #ignore case
            else:
                bbdd_text='empty'
            bbdd_texts[image_id] = bbdd_text


    return bbdd_texts


def get_text(img, text_box):

    tl_x=text_box[0]
    tl_y=text_box[1]
    br_x=text_box[2]
    br_y=text_box[3]

    if text_box!=[0, 0, 0, 0]:
        roi=img[tl_y:br_y,tl_x:br_x]
        gray=cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
        thld,bw=cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        text = tess.image_to_string(bw)
    else:
        text="box not found"

    special_chars = [c for c in set(text).difference(ascii_letters + digits + ' ')]
    text_filtered = ''.join((filter(lambda s: s not in special_chars, text)))
    final_text = ' '.join(text_filtered.split())

    print(f'Text before: {text} --> Text filtered: {final_text}')
    return final_text

def get_text_distance(text_1,text_2,distance_metric="Levensthein"):
    if distance_metric=="Levensthein":
        distance=jel.levenshtein_distance(text_1,text_2)

    elif distance_metric=="Hamming":
        distance=jel.hamming_distance(text_1,text_2)

    elif distance_metric=="Damerau":
        distance=jel.damerau_levenshtein_distance(text_1,text_2)

    else:
        print('Metric doesn\'t exist')
    return distance

def get_k_images(painting, text_box, bbdd_texts, k=10, distance_metric="Hamming"):

    text = get_text(painting, text_box)
    distances = {}

    for bbdd_id, bbdd_text in bbdd_texts.items():

        if bbdd_text!='empty':
            bbdd_text=bbdd_text.replace("(","").replace("'"," ").replace(")","")
            distances[bbdd_id] = get_text_distance(text.lower(), bbdd_text.split(",",1)[0].strip(),distance_metric)

        else:
            distances[bbdd_id]=100

    min_distance = min(distances.values())
    author_images = [key for key in distances if distances[key] == min_distance]
    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=False))[:k]


    return [predicted_image[0] for predicted_image in k_predicted_images], author_images,distances
