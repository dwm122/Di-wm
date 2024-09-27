import numpy as np
import os
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from osgeo import gdal


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + FN)  
    a = np.diag(confusionMatrix).sum()
    b = confusionMatrix.sum()
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def read_img(filename):
    dataset = gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def eval_new(label_all, predict_all, classNum):
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    confusionMatrix = confusion_matrix(predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    print("")
    print("混淆矩阵:")
    print(confusionMatrix)
    print("精确度:")
    print(precision)
    print("召回率:")
    print(recall)
    print("F1-Score:")
    print(f1ccore)
    print("整体精度:")
    print(OA)
    print("IoU:")
    print(IoU)
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)
    return confusionMatrix



if __name__ == "__main__":
    #################################################################
    #  标签图像文件夹
    LabelPath = r"/root/autodl-tmp/mycode/ASPP-unet/dataset/test_label"
    #  预测图像文件夹
    PredictPath = r"/root/autodl-tmp/mycode/ASPP-unet/dataset/result/val8"
    log_path = r"/root/autodl-tmp/mycode/ASPP-unet/混淆矩阵/ASPP+通道注意力（参数16 学习率0.001）.csv"

    for path in [PredictPath, os.path.dirname(log_path)]:
        if not os.path.exists(path):
            os.mkdir(path)

    #  类别数目(包括背景)
    classNum = 7

    #  获取文件夹内所有图像
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    #  读取第一个图像，后面要用到它的shape
    # Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)
    im_proj, im_geotrans, im_width, im_height, Label0 = read_img(LabelPath + "//" + labelList[0])

    #  图像数目
    label_num = len(labelList)

    #  把所有图像放在一个数组里
    label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
    for i in range(label_num):
        im_proj, im_geotrans, im_width, im_height, Label = read_img(LabelPath + "//" + labelList[i])
        label_all[i] = Label

        im_proj, im_geotrans, im_width, im_height, Predict = read_img(PredictPath + "//" + PredictList[i])
        predict_all[i] = Predict

    confusionMatrix = eval_new(label_all, predict_all, classNum)
    np.savetxt(log_path, confusionMatrix, delimiter=",")
