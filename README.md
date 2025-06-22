# 运行步骤：

## 模型训练:

1、load_data_2025.py # 加载CPSC2025的训练数据

2、train_evaluate_2017.py # 在PhysioNet2017 数据集上进行训练，得到预训练模型

3、finetune.py # 模型微调得到最后模型权重

## 测试集预测

4、get_csv.py # 使用训练好的模型在CSPC2025测试数据集上进行测试，得到results.csv。
	results.csv中，第一列为测试集数据索引，第二列为预测标签，N表示正常，AF表示房颤
