from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/manisha/Desktop/yolo/repo/players/players.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/manisha/Desktop/yolo/repo/players/players/images/test/7_webp.rf.7997c662e4898affbd51812eff1532ec.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
