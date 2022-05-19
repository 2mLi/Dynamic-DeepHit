# Changelog

## 0.0.0.5: 2022/05/13

### 添加了......

- `Model_Learning_Curves.py`：将模型拟合时保存的c-index和损失学习曲线可视化。
- `Model_Temporal_Attention.py`：根据（Lee, et. al., 2019）文中所提，分析神经网络对纵向数据的“哪一部分”保持“关注”。在控制台输出long-term和short-term患者数量，并将short-term患者名单（以ID列表的形式）输出，以供其他程序或脚本辅助分析。

## 0.0.0.4: 2022/05/06

### 修复了……

- 在仅使用一（1）个或是零(0)个连续变量拟合模型时，`Model_Patient_Trajectory.py`因`TypeError: 'AxesSubplot' object is not subscriptable`错误而跳出的问题；现在仅使用一个连续变量拟合模型时，此脚本仍能生成正常的连续变量趋势图；不使用连续变量时，脚本会打印提示并跳过此步（因无图可生成）。
- 在变量名过长时，`Model_tvROC.py`生成的图表中，文字超过图表边际的问题；
- `Model_tvROC.py`生成的图表的x，y轴范围因数据问题而有时无法正确生成的问题。现在无论数据的分布如何，此脚本生成的ROC图标中，1-特异性和灵敏度轴的范围都会正确显示为0~1。为了确保正确的图表范围被完整显示，代码中实际使用的范围为0~1.05。

## 0.0.0.3: 

### 添加了……

- `Model_TVROC.py`：计算时间依存ROC与AUC；
- `Model_Patient_Trajectory.py`：计算患者连续纵向变量随时间变化的趋势，并同时展示模型预测患者转癌风险随时间的变化；
- `Model_Dynamic_KMP.py`：在每个时间窗口-位点组合，使用log-rank假设测定寻找最优阈值将患者分为两组，并绘制对应的时间依存Kaplan-Meier曲/折线。

## 0.0.0.2: 

### 添加了……

- `Model_Training.py`：根据给定参数组合，进行统一模型拟合，并将拟合结果保存至本地硬盘以备随时的未来调用。
- `Model_evaluation.py`：基于`Model_Training.py`的训练时给定的时间窗口-位点组合，在每个组合计算时间依存Harrell's c-index和Brier-score，以衡量模型预测患者转癌能力。