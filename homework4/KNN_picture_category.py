import numpy as np
from os import listdir
from PIL import Image
import shutil

pngpath = "./png_data" #存放手写体数字图片
txtpath = "./txt_data" #保存转换矩阵
trainpath =  "./train_data/"
testpath = "./test_data/"

def knn(k, testdata, traindata, labels):
    '''定义算法'''
    traindatasize = traindata.shape[0]
    dif = np.tile(testdata,(traindatasize,1)) - traindata
    sqrdif = dif**2
    sumsqrdif = sqrdif.sum(axis=1)
    distance = sumsqrdif**0.5
    sorted_distance = distance.argsort()
    count = {}
    for i in range(0,k):
        vote = labels[sorted_distance[i]]
        count[vote] = count.get(vote,0)+1
    sorted_count = sorted(count.items(),key=lambda x:x[1],reverse=True)
    return sorted_count[0][0]

def png2txt(pngpath,txtpath):
    '''将固定尺寸的手写体图片转换成0,1数组矩阵文件'''
    pnglist = listdir(pngpath)
    for i in pnglist:
        fname = i.split(".")[0]
        im = Image.open(pngpath+"/"+i)
        fh = open(txtpath+"/"+fname+".txt","a")
        for m in range(0,32):
            for n in range(0,32):
                pix = im.getpixel((n,m))
                pixs = pix[0]+pix[1]+pix[2]
                if pixs==0:
                    fh.write("1")
                else:
                    fh.write("0")
            fh.write("\n")
        fh.close()

def shutildata(txtpath,trainpath,testpath):
    '''将后缀为15的文件分拣出来作为测试集，其他文件为训练集'''
    txtlist = listdir(txtpath)
    for i in txtlist:
        try:
            if (i.split(".")[0].split("_")[1]=="15"):
                shutil.move(txtpath+"/"+i,testpath)
            else:
                shutil.move(txtpath+"/"+i,trainpath)
        except:
            pass

#加载数据
def data2array(fname):
    arr = []
    fh = open(fname)
    for i in range(0,32):
        thisline = fh.readline()
        for j in range(0,32):
            arr.append(int(thisline[j]))
    return arr

#建立训练数据
def traindata():
    labels = []
    trainfile = listdir(trainpath)
    trainarr = np.zeros((len(trainfile),1024))
    for i in range(0,len(trainfile)):
        thislabel = trainfile[i].split(".")[0].split("_")[0]
        if len(thislabel)!=0:
            labels.append(int(thislabel))
        trainarr[i,:] = data2array(trainpath+trainfile[i])
    return trainarr,labels

def datatest():
    png2txt(pngpath, txtpath)
    shutildata(txtpath, trainpath, testpath)
    trainarr,labels = traindata()
    testfiles = listdir(testpath)
    for i in range(0,len(testfiles)):
        testpicname = testfiles[i].split("_")[0]
        testarr = data2array(testpath+testfiles[i])
        result = knn(2,testarr,trainarr,labels)
        print("真正数字:"+testpicname+"  "+"测试结果为:{}".format(result))

if __name__ == "__main__":
    datatest()
