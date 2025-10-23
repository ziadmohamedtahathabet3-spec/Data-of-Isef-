# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

def test(image, model_dir, device_id):
    """
    Test if the given image is real or fake.

    Args:
        image (numpy array): The input image.
        model_dir (str): Directory containing the models.
        device_id (int): The device ID to use.

    Returns:
        int: 1 if real, 0 if fake.
    """
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # الحصول على الإطار المحدد حول الوجه
    image_bbox = model_test.get_bbox(image)
    if image_bbox is None:
        print("No face detected.")
        return -1  # في حالة عدم اكتشاف وجه

    prediction = np.zeros((1, 3))
    test_speed = 0

    # التنبؤ باستخدام كل الموديلات
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # الحصول على النتيجة
    label = np.argmax(prediction)
    value = prediction[0][label] / 2

    # طباعة النتيجة للتصحيح
    print(f"Prediction: {prediction}, Label: {label}, Value: {value}, Time: {test_speed:.2f}s")
    return label
