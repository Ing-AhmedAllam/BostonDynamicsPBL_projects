from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

#Get current path
cur_dir = os.path.dirname(os.path.abspath(__file__))

# Train the model
results = model.train(data= cur_dir+"/yolo_train/data.yaml", epochs=100, imgsz=640)