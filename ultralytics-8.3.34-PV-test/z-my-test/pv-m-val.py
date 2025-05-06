from ultralytics import YOLO

if __name__ == '__main__':
    #加载模型
    model_predict = YOLO(r'runs\detect\train\weights\best.pt',task='detect')

    #评价
    result = model_predict.val(
        data=r'D:\YOLOV8\yolo-all\ultralytics-8.3.34-PV-test\datasets\BrokenSolarPanelDetection_test\data.yaml',
        batch=32,
        plots=True
    )
'''

'''