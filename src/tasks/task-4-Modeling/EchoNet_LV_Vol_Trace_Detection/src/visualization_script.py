# visualization_script.py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

def VisualizeSampleImages(img, keypoints, col='#16a085'):
    fig, axs = plt.subplots(1, len(img), figsize=(20,20))
    for i in range(len(img)):
        axs[i].imshow(img[i], cmap='gray')
        axs[i].scatter(keypoints[i][0::2], keypoints[i][1::2], color=col)
    plt.show()

def VisualizeSingleData(data_images, data_keypoints, pred_keypoints, data_ids, index):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(data_images[index], cmap='gray')
    axs[0].scatter(data_keypoints[index][0::2], data_keypoints[index][1::2], color='red')
    axs[0].set_title(f"Actual keypoints for {data_ids[index]}")

    axs[1].imshow(data_images[index], cmap='gray')
    axs[1].scatter(pred_keypoints[index][0::2], pred_keypoints[index][1::2], color='blue')
    axs[1].set_title(f"Predicted keypoints for {data_ids[index]}")

    plt.show()

def Accuracy_ConfusionMatrix(actual, predicted, categories):
    print("Accuracy: ", accuracy_score(actual, predicted))

    cm = confusion_matrix(actual, predicted, labels=categories)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def visualize_error(data_images, data_keypoints, predicted_keypoints, data_ids, df):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Keypoint error
    axs[0].hist(df['AverageErrorPerImage'], bins=50)
    axs[0].set_title('Keypoint error')
    axs[0].set_xlabel('Average error per image')
    axs[0].set_ylabel('Count')

    # Class error
    class_error = 1 - df['ImageClassEqual']
    axs[1].hist(class_error, bins=50)
    axs[1].set_title('Class error')
    axs[1].set_xlabel('Class error per image')
    axs[1].set_ylabel('Count')

    plt.show()

