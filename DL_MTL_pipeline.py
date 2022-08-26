import pandas as pd
import os
import keras
from sklearn import preprocessing
from PredictionModel import PredictionModel
from ResultSave import ResultSave
from ResultAnalysisAndPlot import ResultAnalysisAndPlot

"""------------------------------------------------ 1. 准备工作 -------------------------------------------------"""

# Generate the required folders
path = os.getcwd()
if not os.path.exists(path + '\\results'):
    os.makedirs(path + '\\results')  # save original result
if not os.path.exists(path + '\\final'):
    os.makedirs(path + '\\final')  # save final result
if not os.path.exists(path + '\\mean_files'):
    os.makedirs(path + '\\mean_files')  # save the mean-iterations result
if not os.path.exists(path + '\\model'):
    os.makedirs(path + '\\model')  # save model
if not os.path.exists(path + '\\plot'):
    os.makedirs(path + '\\plot')  # save plot
if not os.path.exists(path + '\\plot\\选择概率'):
    os.makedirs(path + '\\plot\\选择概率')
if not os.path.exists(path + '\\plot\\选择概率拆分'):
    os.makedirs(path + '\\plot\\选择概率拆分')
if not os.path.exists(path + '\\plot\\选择概率稳定性'):
    os.makedirs(path + '\\plot\\选择概率稳定性')

DL_data = pd.read_csv("data/DL_data_final_non_standard.csv")

# mode
Y_m = pd.get_dummies(DL_data["MainTransport"]).astype('int')  # one-hot
# purpose
Y_p = pd.get_dummies(DL_data["Travel_Purpose"]).astype('int')  # one-hot

# remove unused columns
DL_data.drop(columns=['Unnamed: 0', 'index', 'Family_NO', 'Person_NO', 'Travel_Purpose', 'MainTransport',
                      'Career_Code', 'Family_Region'
                      ], inplace=True)

DL_data.drop(columns=[
    # 'Departure_longitude', 'Departure_latitude', 'Destination_longitude', 'Destination_latitude',
    'Dayoff_Mon', 'Dayoff_Wed', 'Dayoff_Thur', 'Dayoff_Fri',
], inplace=True)

X_train_raw = DL_data  # X_train_raw --> Used for plot
X = DL_data.copy()

input_size = len(X.columns)  # input_size
all_variables = list(X.columns)
'''
    all_variables :
    
    'Departure_Time', 'Arrival_Time', 'trip_time','distance', 'SumCost',  
    'Departure_longitude', 'Departure_latitude', 'Destination_longitude', 'Destination_latitude',
    'Cars_Count', 'Sex', 'Age', 'Education', 'Is_Driver', 
    'Dayoff_Tue', 'Dayoff_Sat', 'Dayoff_Sun'
    'Total_IN', 'Area', 'Is_Own_House', 'Members_Count',
    'Region_ZhongShan', 'Region_XiGang', 'Region_ShaHeKou', 'Region_GanJingZi', 'Region_GaoXinYuan', 'Region_LvShun',
    'Region_JinPu', 'Region_PuLanDian', 'Region_WaFangDian', 'Region_ZhuangHe', 'Region_ChangHai', 
    'Career_Woker', 'Career_Farmer', 'Career_Administrative', 'Career_Primary', 'Career_Junior', 'Career_Unemployed', 
    'Career_Service', 'Career_Education', 'Career_Doctor', 'Career_Technical', 'Career_Individual', 'Career_Soldier',
    'Career_Freelancer', 'Career_Retiree', 'Career_Housewife', 'Career_Others'
'''

cont_variables = ['Departure_Time', 'Arrival_Time','trip_time', 'distance', 'SumCost',
                  'Departure_longitude', 'Departure_latitude', 'Destination_longitude', 'Destination_latitude'
                  ]

"""--------------------- 是否对连续变量进行归一化 ---------------------"""

# CONT_VARIABLES_STANDARDIZATION = True
CONT_VARIABLES_STANDARDIZATION = False

if CONT_VARIABLES_STANDARDIZATION is True:
    df_cont_var = pd.DataFrame()
    for col in cont_variables:
        df_cont_var[col] = DL_data[col]
    df_cont_var = pd.DataFrame(preprocessing.scale(df_cont_var))
    df_cont_var.columns = cont_variables
    for col in cont_variables:
        X[col] = df_cont_var[col]

# print(X.columns)

"""--------------------- 预测内容控制 ---------------------"""

# 是否进行多任务预测 ? True or False
Run_multitask = True
Analysis_multitask = True
# 是否进行单任务预测 ? True or False
Run_single_task = True
Analysis_single_task = True

# 是否保存网络模型（当确定好最优参数以后可以保存神经网络模型）
SAVE_MODEL = True

# 是否进行分析绘图
ANALYSIS_AND_PLOT = True
# 绘图选择
# 绘制各种交通方式或出行目的随着自变量（出行时间、出行距离、出行费用）增长的变化
PLOT_SUBSTITUTE_PATTERN = True
# 绘制单一交通方式或出行目的随着自变量（出行时间、出行距离、出行费用）增长的变化
PLOT_CHOICE_PROBABILITY = True
# 绘制单一交通方式或出行目的对自变量（出行时间、出行距离、出行费用）的波动敏感性
PLOT_PROBABILITY_DERIVATIVE = True

"""--------------------- 参数调优取值范围 ---------------------"""
# epoch
epoch = 50
# iterations
iterations = 1
# single-task prediction item       # ["mode", "purpose"]
single_pre_item_list = ["purpose"]

# mode coefficient
# coe_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
# coe_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
# coe_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
# coe_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# coe_list = [0.05, 0.1, 0.5, 0.9, 0.95]
# coe_list = [0.05,0.5,0.95]
coe_list = [0, 1]

# number of neurons     # [128, 256]
n_hidden_list = [128]
# learning rate      # [1e−1, 1e−2, 5e−3, 1e−3, 5e−4]
lr_list = [1e-3]
# dropout rate      # [1e−2, 1e−3, 1e−5]
dr_list = [1e-2]
# l1 weight of regular.l1_l2    # [1e−20, 1e−10, 1e−5]
l1_count_list = [1e-20]
# l2 weight of regular.l1_l2    # [1e−20, 1e−10, 1e−5]
l2_count_list = [1e-10]
# batch size        # [128, 256]
batch_size_list = [256]
# list of hidden layers     # [2, 3, 4, 5,6]
hidden_layer_list = [4]
# list of shared_hidden_layers      # [1, 2, 3]
shared_hidden_layers_list = [2]
# list of task specific layers      # [1, 2, 3]
task_specific_layers_list = [2]

# 激活函数      # ['relu','tanh','sigmoid',...]
activation_name = 'relu'

"""----------------------------------------------- 2. 模型 ------------------------------------------------"""

# PredictionModel ---- including single-task prediction and multi-task prediction models
PRE = PredictionModel(X, input_size, Y_m, Y_p, epoch, iterations, coe_list, single_pre_item_list,
                      n_hidden_list, lr_list, dr_list, l1_count_list, l2_count_list, batch_size_list, activation_name,
                      SAVE_MODEL)
# RESULT_ANALYSIS ---- Save the results of each run
RS = ResultSave(epoch, iterations, coe_list, single_pre_item_list, n_hidden_list, lr_list, dr_list,
                l1_count_list, l2_count_list, batch_size_list, shared_hidden_layers_list,
                task_specific_layers_list, hidden_layer_list, Analysis_multitask, Analysis_single_task)

"""--------------------- 单任务预测 ---------------------"""

if Run_single_task is True:
    for hidden_layers in hidden_layer_list:
        # 运行模型
        PRE.singletask_prediction(hidden_layers)
        # 结果无编辑保存
        RS.singletask_result_save(hidden_layers)

# 结果格式化保存
if Analysis_single_task is True:
    RS.stl_result_regular()
"""--------------------- 多任务预测 ---------------------"""

if Run_multitask is True:
    for shared_hidden_layers in shared_hidden_layers_list:
        for task_specific_layers in task_specific_layers_list:
            # 运行模型
            PRE.multitask_prediction(shared_hidden_layers, task_specific_layers)
            # 结果无编辑保存
            RS.multitask_result_save(shared_hidden_layers, task_specific_layers)

# 结果格式化保存
if Analysis_multitask is True:
    RS.mtl_result_regular()

"""--------------------- 结果合并 ---------------------"""
# 单任务多任务预测结果合并
if Analysis_single_task is True and Analysis_multitask is True:
    # save multi-task and single-task results together regularly
    RS.mtl_stl_result_regular()

"""------------------------------------------ 3. 数据可视化 -------------------------------------------"""
'''
compute the probability and probability derivative plot economic information. 
plot including substitute_pattern, choice probability and probability derivative
'''
# 绘图参数
# modes = ['步行','公共汽车','地铁轻轨','出租车或网约车','私家车','定制公交或共享汽车']
# purposes = ['上班','上学','回家','购物','娱乐','旅游或出差','接送家人']
# 自变量  '出行时间 (分钟)', '出行距离 (公里)', '出行费用 (元)'


if ANALYSIS_AND_PLOT is True:

    # coe_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # 分析出行方式，则 mode_coe = 1 分析出行目的则为 0
    mode_coe = 0
    RAAP = ResultAnalysisAndPlot(X, X_train_raw, all_variables, cont_variables, iterations,mode_coe,
                                 PLOT_SUBSTITUTE_PATTERN, PLOT_CHOICE_PROBABILITY, PLOT_PROBABILITY_DERIVATIVE)

    ''' MTL_M(多任务预测-出行方式)、 STL_M(单任务预测-出行方式)、 MTL_P(多任务预测-出行目的)、 STL_P(单任务预测-出行目的) '''

    # RAAP.forward("MTL_M")
    # RAAP.forward("STL_M")
    RAAP.forward("MTL_P")
    RAAP.forward("STL_P")

