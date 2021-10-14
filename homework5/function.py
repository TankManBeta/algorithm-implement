#导入自带数据集
from sklearn import datasets
#导入交叉验证库
from sklearn import model_selection
#导入SVM分类算法库
from sklearn import svm
#导入图表库
import matplotlib.pyplot as plt

#读取自带数据集并赋值给digits
digits = datasets.load_digits()

#绘制图表查看数据集中数字9的图像
plt.imshow(digits.images[9], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('digits.target[9]')
plt.show()

Y=digits.target
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, 64))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.4, random_state=0)

#生成SVM分类模型
clf = svm.SVC(gamma=0.001)
#使用训练集对svm分类模型进行训练
clf.fit(X_train, y_train)
print("模型的准确率为：{}".format(clf.score(X_test, y_test)))

#对测试集数据进行预测
predicted=clf.predict(X_test)
print(predicted[:100])

#查看测试集中前100个真实结果
expected=y_test
print(expected[:100])

#打印识别错误的数字位置
for i in range(100):
    if predicted[i] != expected[i]:
        print("第{}个识别错误".format(i+1))
    else:
        pass