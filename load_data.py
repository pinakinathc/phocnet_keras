'''This loads data in accordance to the standards mentioned in the IAM database.
Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''

from glob import glob
import cv2
from xml.etree import ElementTree as ET
import numpy as np

from create_phoc_label import generate_label
from datetime import datetime

WORD_IMAGE_DIR = 'words/'
XML_DIR = 'xml/'
transcripts = {}


def rule():
    """IAM Dataset has some set of rules against which we must compare
    our models. We are loading those rules to set:
    (Training_data, Validation_data, Test_data)
    """
    with open('rules/trainset.txt', 'r') as fp: # Path to train ids file
        train_rule = fp.readlines()
    train_rule = [x.strip() for x in train_rule]

    with open('rules/validationset1.txt', 'r') as fp: # Path to valid_1 file
        valid_rule = fp.readlines()
    with open('rules/validationset2.txt', 'r') as fp: # Path to valid_2 file
        valid_rule += fp.readlines()
    valid_rule = [x.strip() for x in valid_rule]

    with open('rules/testset.txt', 'r') as fp: # Path to test ids file
        test_rule = fp.readlines()
    test_rule = [x.strip() for x in test_rule]

    return train_rule, valid_rule, test_rule


def append_data(x, y, transcript, data): # need not return anything
    x.append(data[0])
    y.append(data[1])
    transcript.append(data[2])


def load_data():
    time_start = datetime.now()

    train_rule, valid_rule, test_rule = rule()

    xml_files = glob(XML_DIR+'*.xml')

    x_train = []
    y_train = []
    train_transcript = []
    x_valid = []
    y_valid = []
    valid_transcript = []
    x_test = []
    y_test = []
    test_transcript = []
    global transcripts
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Change this code to get the corresponsing word image dir
        image_dir = xml_file.split('/')[-1].split('.')[0].split('-')
        image_dir = image_dir[0] + '/' + image_dir[0]+'-'+image_dir[1]+ '/'
        image_dir = WORD_IMAGE_DIR + image_dir

        for word in root.iter('word'):
            img_id = word.get('id')
            img_name = image_dir+img_id+'.png'
            img_line = '-'.join(img_id.split('-')[:-1])
            img_transcript = word.get('text').lower()

            img = cv2.imread(img_name, 0)
            if img is None: # Some image files are corrupted
                continue

            target = generate_label(img_transcript)
            if sum(target) == 0: # For special characters
                img_transcript = '' # Use a special notation for them

            img = cv2.resize(img, (100, 50))
            img = np.where(img<200, 1, 0)
            img = img[:, :, np.newaxis]

            data = [img, target, img_transcript]

            if img_line in train_rule:
                append_data(x_train, y_train, train_transcript, data)
            elif img_line in valid_rule:
                append_data(x_valid, y_valid, valid_transcript, data)
            elif img_line in test_rule:
                append_data(x_test, y_test, test_transcript, data)

    N = len(x_train) + len(x_valid) + len(x_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_trainscript = np.array(train_transcript)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    valid_transcript = np.array(valid_transcript)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    test_transcript = np.array(test_transcript)

    print ("Time to fetch data: ", datetime.now() - time_start)

    return (x_train, y_train, train_transcript,
            x_valid, y_valid, valid_transcript,
            x_test, y_test, test_transcript)
