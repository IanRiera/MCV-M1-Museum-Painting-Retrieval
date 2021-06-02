import pytesseract as tess
# tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'#your path to tesseract for windows users
tess.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'#your path to tesseract for windows users
# tess.pytesseract.tesseract_cmd = r'/home/oscar/.local/bin/pytesseract'
from PIL import Image
import operator
import cv2 as cv
import jellyfish as jel
import os
import imutils

from string import ascii_letters, digits

def get_bbdd_texts(bbdd_path):
    bbdd_texts = {}
    for image_filename in sorted(os.listdir(bbdd_path)):
        if image_filename.endswith('.txt'):
            image_id = int(image_filename.replace('.txt', '').replace('bbdd_', ''))
            f = open(os.path.join(bbdd_path, image_filename), "r", encoding='latin-1')
            line= f.readline()
            if line.strip():
                bbdd_text = line.lower().replace("(","").replace("'"," ").replace(")","") #ignore case
            else:
                bbdd_text='empty'
            bbdd_texts[image_id] = bbdd_text


    return bbdd_texts

def dilate_text_box(text_box,percentage):
    tl_x=int(text_box[0]*(1-percentage/100))
    tl_y=int(text_box[1]*(1-percentage/100))
    br_x=int(text_box[2]*(1+percentage/100))
    br_y=int(text_box[3]*(1+percentage/100))

    return[tl_x,tl_y,br_x,br_y]


def get_text(img, text_box):
    if text_box!=[0, 0, 0, 0]:
        percentage=2
        expanded_box=dilate_text_box(text_box,percentage)

        tl_x=expanded_box[0]
        tl_y=expanded_box[1]
        br_x=expanded_box[2]
        br_y=expanded_box[3]

        roi=img[tl_y:br_y,tl_x:br_x]
        resized=imutils.resize(roi,height=500)
        gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
        thld,bw=cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        text = tess.image_to_string(bw)

        # cv.imshow("roi",roi)
        # cv.imshow("resized",resized)
        # cv.imshow("bw",bw)
        # cv.waitKey()
    else:
        text="box not found"

    special_chars = [c for c in set(text).difference(ascii_letters + digits + ' ')]
    text_filtered = ''.join((filter(lambda s: s not in special_chars, text)))
    final_text = ' '.join(text_filtered.split())

    # print(f'Text before: {text} --> Text filtered: {final_text}')
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

def compute_distances(paintings, text_boxes, bbdd_texts, metric, weight):

    distances_all = []
    for image_id, paintings_per_img in enumerate(paintings):
        distances_img = []

        text_boxes_image = text_boxes[image_id]

        for painting_id, painting in enumerate(paintings_per_img):
            if len(text_boxes_image) > painting_id:
                text_box = text_boxes_image[painting_id]
            else:
                text_box = [0,0,0,0]
            if text_box is None:
                text_box = [0,0,0,0]

            text = get_text(painting, text_box)
            distances = []

            for bbdd_id, bbdd_text in bbdd_texts.items():

                if bbdd_text!='empty':
                    bbdd_text=bbdd_text.replace("(","").replace("'"," ").replace(")","")
                    distances.append(weight * get_text_distance(text.lower(), bbdd_text.split(",",1)[0].strip(),metric))

                else:
                    distances.append(10000)
            distances_img.append(distances)
        distances_all.append(distances_img)
    return distances_all
