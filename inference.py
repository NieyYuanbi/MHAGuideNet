import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import FirstNet, My2DModel, IntegratedModel  # 从你的模型文件导入模型类
from wrap.ADNI import AdniDataSet  # 从你的数据文件导入数据集类
from wrap.setting import parse_opts
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = parse_opts()
sets.gpu_id = [0]

# 加载已训练的模型
model_2d = My2DModel(f=8)
model_3d = FirstNet(f=8)
state_dict = torch.load('', map_location=torch.device('cpu'))
# 移除 "module." 前缀
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model_3d.load_state_dict(new_state_dict)
model_3d.to(device)

model = IntegratedModel(model_2d, model_3d)
model = torch.nn.DataParallel(model)
model_weight_path = ""  # 模型权重文件路径
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
model.eval()

# 加载测试数据集
test_data_path = ''
test_img_path = r''
test_data = AdniDataSet(test_data_path, test_img_path, sets)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)
print("数据加载成功")

# 迭代测试数据并进行预测
predictions = []
all_label = []
probabilities =[]

with torch.no_grad():
    print("testing......")
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, patch1, patch2, patch3, labels = data
        print("labels:", labels)
        all_label.extend(labels.cpu().numpy())
        inputs, patch1, patch2, patch3, labels = inputs.to(device), patch1.to(device), patch2.to(
            device), patch3.to(device), labels.to(device)

        feature1, feature2, feature3 = model_2d(patch1, patch2, patch3)
        feature3d = model_3d(inputs)
        output = model(feature1, feature2, feature3, feature3d)
        predict = torch.softmax(output, 1)[:, 1]
        probabilities.extend(predict.cpu().numpy())
        print("probabilities:", probabilities)
        predict[predict >= 0.6] = 1
        predict[predict < 0.6] = 0
        predictions.extend(predict.cpu().numpy())
        print("predictions:", predictions)

csvfile = r""
correct_predictions = 0

with open(csvfile, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['probabilities', 'predictions', 'label'])

    for probability, prediction, label in zip(probabilities, predictions, all_label):
        csvwriter.writerow([probability, prediction, label])

        if prediction == label:
            correct_predictions += 1

accuracy = correct_predictions / len(predictions)
print("Accuracy:", accuracy)

# 将预测和标签转换为numpy数组
predictions = np.array(predictions)
all_label = np.array(all_label)
probabilities = np.array(probabilities)

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(all_label, predictions).ravel()

# 计算各项指标
acc = accuracy_score(all_label, predictions)
sen = recall_score(all_label, predictions)  # 敏感性
spe = tn / (tn + fp)  # 特异性
pre = precision_score(all_label, predictions)  # 精确度
f1 = f1_score(all_label, predictions)  # F1分数
AUC = roc_auc_score(all_label, probabilities)

# 打印结果
print("Accuracy:", acc)
print("Sensitivity:", sen)
print("Specificity:", spe)
print("Precision:", pre)
print("F1 Score:", f1)
print("Auc", AUC)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(all_label, probabilities)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("ROC AUC:", roc_auc)


