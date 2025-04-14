from ultralytics import YOLO
import os

#get current path
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cur_dir, "runs/segment/train2/weights/best.pt")
# Load a custom model
model = YOLO(model_path)  # load a custom model (recommended for inference)

# Predict with the model
results = model.predict(source=cur_dir+"/yolo_train/dataset/test/images", conf=0.25, show=True, save=True)  # predict on an image

# Print results
print(results)  # print results to screen
print("Results saved to: ", results[0].save_dir)  # print results to screen
# Print results to file
# results[0].save()  # save results to file

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(cur_dir+'/runs/segment/train2/results.csv')

# List of columns to plot
columns_to_plot = ['train/seg_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

# Create a separate plot for each column
for column in columns_to_plot:
    df[[column]].plot()
    plt.title(f'{column} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
