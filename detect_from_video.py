"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import glob
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#import models
#import xception
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import matplotlib.pyplot as plt
import pandas as pd

preds = []
true = []

d={
    1:"xception",
    2:"efficientnetv2"
            }
a=2
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)
    #print(output)
    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    #print('Starting: {}'.format(video_path))
    total_frames=0
    # print("model_path: ",model_path)
    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.mp4'
    if video_path.split('/')[-1] == '.DS_Store':
        #print("Lol")
        return
    # print(video_path)
    # if 'real' in video_path:
    #     true.append(0)
    # else:
    #     true.append(1)
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    #print("printing the number of frames: ", num_frames)
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    
    model, *_ = model_selection(modelname=d[a], num_out_classes=2)
    #clearprint("out of model.py")
    #print(model)
    
    if model_path is not None:
        model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)
    real = 0
    fake = 0
    # print(video_path)
    if 'real' in video_path:
        true.append(0)
    else:
        true.append(1)
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)
        
        total_frames=total_frames+1
        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------
            # if 'real' in video_path:
            #     print("True Label", 0)
            # else:
            #     print("True Label", 1)
            # print("pritning the prediction output: ",prediction)
            # print("output: ",output)
            # preds.append(prediction)
            if prediction == 0:
                real=real+1
            else:
                fake=fake+1

            # Draw bounding box
            
            # return
            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'real' if prediction == 0 else 'fake'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        # Show
        cv2.imshow('test', image)
        cv2.waitKey(33)     # About 30 fps
        writer.write(image)
        
    pbar.close()
    if real>=fake:
        preds.append(0)
    else:
        preds.append(1)
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    # print(video_path)
    a = 1
    
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        a=int(input("Select a model: "))
        test_full_image_network(**vars(args))
    else:
        videos = glob.glob(video_path+'/**/*.mp4')
        # print(videos)
        #videos = os.listdir(video_path)
        #print("printing length of the folder:",len(videos))
        a=int(input("Select a model: "))
        #print(d[a])cl
        for video in videos:
            args.video_path = join(video_path, video)
            #print("printing the args vars: ", **vars(args)
            total_frames=0
            test_full_image_network(**vars(args))
    #print("printing true list values:\n", true)
    #print("printing pred list values:\n", preds)
    if len(true)<len(preds):
        true=preds[:len(true)]

    
    print("Accuracy:", accuracy_score(true, preds))
    print("Precision:", precision_score(true, preds))
    print("F1 Score:", f1_score(true, preds))
    print("Recall:", recall_score(true, preds))
    a_1 = accuracy_score(true, preds)
    p_1 = precision_score(true, preds)
    r_1 = recall_score(true, preds)
    f_1 = f1_score(true, preds)
    metrics_1 = {"Accuracy": {1:accuracy_score(true, preds)}, "Precision": {1:precision_score(true, preds)}, "F1 Score": {1:f1_score(true, preds)}, "Recall": {1:recall_score(true, preds)}}
    true = []
    preds = []
    a = 2
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = glob.glob(video_path+'/**/*.mp4')
        # print(videos)
        #videos = os.listdir(video_path)
        #print("printing length of the folder:",len(videos))
        a=int(input("Select a model: "))
        #print(d[a])cl
        for video in videos:
            args.video_path = join(video_path, video)
            #print("printing the args vars: ", **vars(args)
            total_frames=0
            test_full_image_network(**vars(args))
    #print("printing true list values:\n", true)
    #print("printing pred list values:\n", preds)
    if len(true)<len(preds):
        true=preds[:len(true)]

    print("Accuracy:", accuracy_score(true, preds))
    print("Precision:", precision_score(true, preds))
    print("F1 Score:", f1_score(true, preds))
    print("Recall:", recall_score(true, preds))

    # print(true, preds)

    a_2 = accuracy_score(true, preds)
    p_2 = precision_score(true, preds)
    r_2 = recall_score(true, preds)
    f_2 = f1_score(true, preds)
    metrics_2 = {"Accuracy": {"XceptionNet":a_1, "EfficientNetV2":a_2}, "Precision": {"XceptionNet":p_1, "EfficientNetV2":p_2}, "F1 Score": {"XceptionNet":f_1, "EfficientNetV2":f_2}, "Recall": {"XceptionNet":r_1, "EfficientNetV2":r_2}}
    df_1 = pd.DataFrame(metrics_1)
    df_2 = pd.DataFrame(metrics_2)

    fig, ax = plt.subplots()
    df_2.plot(ax=ax, kind='bar', legend=False)
    ax.set_xticklabels(df_2.index, rotation=0)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('results.png', bbox_inches='tight')
    # df_1.plot(kind='bar')
    plt.show()
    plt.savefig('plot.png', bbox_inches='tight')
    # print(true, preds)
    # df_1.write_csv("Xception.csv")
    # df_2.write_csv("Efficient.csv")