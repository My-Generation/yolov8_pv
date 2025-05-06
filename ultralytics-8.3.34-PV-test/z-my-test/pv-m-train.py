from ultralytics import YOLO

if __name__ == '__main__':
    #采用m权重文件
    model = YOLO('yolov8m.pt')

    #训练模型,代码运行workers设置0,yaml文件内存放了数据集的路径和数据的标签类型及数据,数据集太小无法得出结果
    model.train(data=r'D:\YOLOV8\yolo-all\ultralytics-8.3.34-PV-test\datasets\BrokenSolarPanelDetection_test\data.yaml',workers=0,epochs=50,batch=16)

'''
Ultralytics 8.3.34 🚀 Python-3.9.19 torch-2.5.1+cu121 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
Model summary (fused): 218 layers, 25,842,655 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:05<00:00,  1.24it/s]
                   all        206       1062      0.903      0.883      0.913      0.654
                broken         24         69      0.687      0.731       0.73      0.424
                 dirty         33         45      0.975      0.956      0.961      0.774
               powdery        110        311      0.945      0.859      0.927      0.634
              stronger         41         41      0.947      0.976      0.992      0.824
                  leaf        101        596      0.964      0.896      0.956      0.616
Speed: 0.2ms preprocess, 21.5ms inference, 0.0ms loss, 0.4ms postprocess per image
'''