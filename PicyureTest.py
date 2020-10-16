
from __future__ import division
import os
import threading
from imageai.Prediction import ImagePrediction
from imageai.Prediction.Custom import ModelTraining
from imageai.Prediction.Custom import CustomImagePrediction

def modelIrain(dataDir='data',classNum=2,epochs=100,batch_size=32):
    '''
    模型训练部分
    '''
    #创建了ModelTraining类的新实例
    model_trainer = ModelTraining()
    #将模型类型设置为ResNet
    model_trainer.setModelTypeAsResNet()
    #设置我们想要训练的数据集的路径
    model_trainer.setDataDirectory(dataDir)
    #模型训练
    '''
    num_objects:该参数用于指定图像数据集中对象的数量
    num_experiments:该参数用于指定将对图像训练的次数，也称为epochs
    enhance_data(可选):该参数用于指定是否生成训练图像的副本以获得更好的性能
    batch_size:该参数用于指定批次数量。由于内存限制，需要分批训练，直到所有批次训练集都完成为止。
    show_network_summary:该参数用于指定是否在控制台中显示训练的过程
    '''
    model_trainer.trainModel(num_objects=classNum,
                             num_experiments=epochs,enhance_data=True,
                             batch_size=batch_size,
                             show_network_summary=True)
    print('Model Train Finished!!!')

    def modelPredict(model_path='data/models/model_ex-001_acc-0.500000.h5',
                     class_path='data/json/model_class.json',
                     pic_path='a.jpg',classNum=2,resNum=5):
        '''

        模型预测部分
        prediction_speed[模型加载的速度]:fast faster fastest
        '''
        prediction=CustomImagePrediction()
        prediction.setModelTypeAsResNet()
        prediction.setModelPath(model_path)
        prediction.setJsonPath(class_path)
        prediction.loadModel(num_objects=classNum,prediction_speed='fastest')
        prediction,probabilities=prediction.predictImage(pic_path,result_count=resNum)
        for eachPrediction, eachProbability in zip(predictions,probabilities):
            print(eachPrediction+" : "+str(eachProbability))

    if __name__=='__main__':
        #模型训练
        modelTrain(dataDir='data',classNum=2,epochs=10,batch_size=8)
        #模型识别预测
        modelPredict(model_path='data/models/model_ex-001_acc-0.500000.h5',
                     class_path='data/json/model_class.json',
                     pic_path='test.jpg',classNum=2,resNum=5)
