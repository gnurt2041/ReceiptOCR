
import os
import joblib
import pickle
import torch
import cv2
import argparse
import numpy as np
from PIL import Image

from segmentation import load_model, extract_base, extract
from alignment import AngleClassificationCNN, transform
from detection import crop_and_transform
from paddleocr import PaddleOCR
from recognition import Predictor
from vietocr.tool.config import Cfg
from pyvi import ViTokenizer

import logging
from ppocr.utils.logging import get_logger as ppocr_get_logger
ppocr_get_logger().setLevel(logging.ERROR)

def get_args():
    parser = argparse.ArgumentParser(description="Receipt OCR")
    parser.add_argument("--image_path", type=str, help="Path to the image")
    return parser.parse_args()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Segmentation module
    # CHECKPOINT_MODEL_PATH = r"./weights/model_mbv3_iou_mix_2C049.pth"
    CHECKPOINT_MODEL_PATH = "./weights/model_r50_iou_mix_2C020.pth"
    trained_model = load_model(num_classes=2, model_name="r50", checkpoint_path=CHECKPOINT_MODEL_PATH, device=device)

    # Alignment module
    model_angle = AngleClassificationCNN().to(device)
    model_angle.load_state_dict(torch.load('./weights/best_angle_classification_cnn.pt'))
    
    # Detection module
    ocr = PaddleOCR(lang="en",show_log=False)

    # Recognition module
    config = Cfg.load_config_from_file('config_vietocr.yml')
    detector = Predictor(config)

    # Classification module
    vectorizer = joblib.load('./weights/vectorizer.pkl')
    f = open('./weights/naive_bayes.pickle', 'rb')
    model_NB = pickle.load(f)
    f.close()

    args = get_args()
    img_path = args.image_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: '{img_path}' could not be located. \n\t Please check the file path and ensure the file exists.")
        exit(1)
    args = get_args()
    img_path = args.image_path

    if os.path.isfile(img_path):
        # Single image path
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]

        document, img_binary = extract_base(image_true=image, trained_model=trained_model)

        image = Image.fromarray((document).astype(np.uint8))
        sample_inputs = transform(image).unsqueeze(0)
        sample_inputs = sample_inputs.to(device)

        model_angle.eval()
        with torch.no_grad():
            predicted, feature_maps = model_angle(sample_inputs)
        _, predict = torch.max(predicted, 1)
        label_name = np.array([0, 90, 270])
        degree = label_name[predict.item()]

        if degree !=0:
            image = np.rot90(np.array(image),4-degree//90)
        else:
            image = np.array(image)
        result = ocr.ocr(image)
        img_PIL = Image.fromarray(image)
        for idx, line in enumerate(result[0]):
            bbox = line[0]
            text = line[1][0]
            score = line[1][1]
            img_bbox = Image.fromarray(crop_and_transform(image,bbox))
            s = detector.predict(img_bbox, return_prob=False)
            vec_text = ViTokenizer.tokenize(s)
            vec_text = vectorizer.transform([vec_text])
            class_text = model_NB.predict(vec_text)[0]
            print(f'{s}: {class_text}')
    else:
        # Directory image path
        for filename in os.listdir(img_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print(f"Processing {filename}...")
                image_path = os.path.join(img_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1]

                document, img_binary = extract_base(image_true=image, trained_model=trained_model)

                image = Image.fromarray((document).astype(np.uint8))
                sample_inputs = transform(image).unsqueeze(0)
                sample_inputs = sample_inputs.to(device)

                model_angle.eval()
                with torch.no_grad():
                    predicted, feature_maps = model_angle(sample_inputs)
                _, predict = torch.max(predicted, 1)
                label_name = np.array([0, 90, 270])
                degree = label_name[predict.item()]

                if degree !=0:
                    image = np.rot90(np.array(image),4-degree//90)
                else:
                    image = np.array(image)
                result = ocr.ocr(image)
                img_PIL = Image.fromarray(image)
                for idx, line in enumerate(result[0]):
                    bbox = line[0]
                    text = line[1][0]
                    score = line[1][1]
                    img_bbox = Image.fromarray(crop_and_transform(image,bbox))
                    s = detector.predict(img_bbox, return_prob=False)
                    vec_text = ViTokenizer.tokenize(s)
                    vec_text = vectorizer.transform([vec_text])
                    class_text = model_NB.predict(vec_text)[0]
                    print(f'{s}: {class_text}')
    