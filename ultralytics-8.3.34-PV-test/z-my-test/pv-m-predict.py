from ultralytics import YOLO

model_predict = YOLO(r'runs\detect\train\weights\best.pt',task='detect')

result_test = model_predict.predict(source=r'D:\YOLOV8\yolo-all\ultralytics-8.3.34-PV-test\datasets\BrokenSolarPanelDetection_test\valid\images'
                                    ,task='detect'
                                    ,save=True)

