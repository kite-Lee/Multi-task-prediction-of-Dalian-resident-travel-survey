import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ResultSave:
    """analysis results"""

    def __init__(self, epoch, iterations, coe_list, single_pre_item_list, n_hidden_list, lr_list, dr_list,
                 l1_count_list, l2_count_list, batch_size_list, shared_hidden_layers_list, task_specific_layers_list,
                 hidden_layer_list, Analysis_multitask, Analysis_single_task):

        self.epoch = epoch
        self.iterations = iterations
        self.coe_list = coe_list
        self.prediction_item = single_pre_item_list
        self.n_hidden_list = n_hidden_list
        self.lr_list = lr_list  # 1e-3
        self.dr_list = dr_list  # 1e-2
        self.l1_count_list = l1_count_list  # 1e-5
        self.l2_count_list = l2_count_list  # 1e-3
        self.batch_size_list = batch_size_list  # 256

        self.shared_hidden_layers_list = shared_hidden_layers_list
        self.task_specific_layers_list = task_specific_layers_list
        self.hidden_layer_list = hidden_layer_list

        self.Analysis_multitask = Analysis_multitask
        self.Analysis_single_task = Analysis_single_task

    def multitask_result_save(self, shared_hidden_layers, task_specific_layers):
        self.shared_hidden_layers = shared_hidden_layers
        self.task_specific_layers = task_specific_layers

        # 计算均值方差
        for coe in self.coe_list:

            for n_hidden in self.n_hidden_list:
                for lr in self.lr_list:
                    for dr in self.dr_list:
                        for l1_count in self.l1_count_list:
                            for l2_count in self.l2_count_list:
                                for batch_size in self.batch_size_list:
                                    csv_list = []
                                    for iters in range(self.iterations):
                                        inputfile = "results/result_" + str(coe) + "_" + \
                                                    str(self.shared_hidden_layers) + "_" + str(self.task_specific_layers) + "_" + str(n_hidden) + "_" + \
                                                    str(l1_count) + "_" + str(l2_count) + "_" + str(lr) + "_" + str(
                                            dr) + "_" + str(batch_size) + \
                                                    "_" + str(iters) + ".csv"
                                        csv_list.append(inputfile)  # 将每一次的迭代文件名放在同一个列表
                                    # print(csv_list)  # 打印原始数据列表，校验
                                    outputfile = "result_" + str(coe) + "_" + str(self.shared_hidden_layers) + "_" + \
                                                 str(self.task_specific_layers) + "_" + str(n_hidden) + "_" + str(
                                        l1_count) + "_" + str(l2_count) + "_" + \
                                                 str(lr) + "_" + str(dr) + "_" + str(batch_size) + ".csv"

                                    filepath = csv_list[0]
                                    print(filepath)
                                    df = pd.read_csv(filepath)
                                    df = df.to_csv("results/" + outputfile, index=False)
                                    for i in range(1, len(csv_list)):
                                        filepath = csv_list[i]
                                        df = pd.read_csv(filepath)
                                        df = df.to_csv("results/" + outputfile, index=False, header=False, mode='a+')

                                    # mean and std
                                    data = pd.read_csv("results/" + outputfile)
                                    data = data.apply(pd.to_numeric, errors='ignore')
                                    df_groupby = data.groupby(['epoch'], as_index=True)
                                    df_groupby.mean().to_csv("mean_files/mean_" + outputfile, index=True)  # 保存均值
                                    # df_groupby.std().to_csv("std_files/std_" + outputfile, index=True)  # 保存标准差

                                    outputfile = "result_" + str(coe) + "_" + \
                                                 str(self.shared_hidden_layers) + "_" + str(self.task_specific_layers) + \
                                                 "_" + str(n_hidden) + "_" + str(l1_count) + "_" + str(l2_count) + "_" + \
                                                 str(lr) + "_" + str(dr) + "_" + str(batch_size) + ".csv"

    def singletask_result_save(self, hidden_layers):

        self.hidden_layers = hidden_layers

        for pre_name in self.prediction_item:
            for n_hidden in self.n_hidden_list:
                for lr in self.lr_list:
                    for dr in self.dr_list:
                        for l1_count in self.l1_count_list:
                            for l2_count in self.l2_count_list:
                                for batch_size in self.batch_size_list:
                                    csv_list = []
                                    for iters in range(self.iterations):
                                        inputfile = "results/result_" + str(
                                            pre_name) + "_" + str(self.hidden_layers) + "_" + str(n_hidden) + "_" + \
                                                    str(l1_count) + "_" + str(l2_count) + "_" + str(lr) + "_" + str(
                                            dr) + "_" + str(batch_size) + "_" + str(iters) + ".csv"
                                        csv_list.append(inputfile)
                                    # print(csv_list)
                                    outputfile = "result_" + str(pre_name) + "_" + str(
                                        self.hidden_layers) + "_" + str(n_hidden) + "_" + \
                                                 str(l1_count) + "_" + str(l2_count) + "_" + str(lr) + "_" + str(
                                        dr) + "_" + str(batch_size) + ".csv"
                                    filepath = csv_list[0]
                                    print(filepath)
                                    df = pd.read_csv(filepath)
                                    df = df.to_csv("results/" + outputfile, index=False)

                                    # 合并每一次迭代的csv文件
                                    for i in range(1, len(csv_list)):
                                        filepath = csv_list[i]
                                        df = pd.read_csv(filepath)
                                        df = df.to_csv("results/" + outputfile, index=False, header=False, mode='a+')
                                    data = pd.read_csv("results/" + outputfile)
                                    data = data.apply(pd.to_numeric, errors='ignore')
                                    df_groupby = data.groupby(['epoch'], as_index=True)
                                    df_groupby.mean().to_csv("mean_files/mean_" + outputfile, index=True)
                                    # df_groupby.std().to_csv("std_files/std_" + outputfile, index=True)

    def mtl_result_regular(self):
        multi_file = []
        for coe in self.coe_list:
            for shared_hidden_layers in self.shared_hidden_layers_list:
                for task_specific_layers in self.task_specific_layers_list:
                    for n_hidden in self.n_hidden_list:
                        for lr in self.lr_list:
                            for dr in self.dr_list:
                                for l1_count in self.l1_count_list:
                                    for l2_count in self.l2_count_list:
                                        for batch_size in self.batch_size_list:
                                            multi_outputfile = "mean_files/mean_result_" + \
                                                               str(coe) + "_" + str(shared_hidden_layers) + \
                                                               "_" + str(task_specific_layers) + "_" + str(
                                                n_hidden) + "_" + str(l1_count) + "_" + str(l2_count) + "_" + \
                                                               str(lr) + "_" + str(dr) + "_" + str(
                                                batch_size) + ".csv"
                                            multi_df = pd.read_csv(multi_outputfile)
                                            multi_rows = len(multi_df) - 1
                                            # print("multi_epoch: ", multi_rows)
                                            a = []
                                            a.append(str(coe))
                                            a.append(str(shared_hidden_layers))
                                            a.append(str(task_specific_layers))
                                            a.append(str(n_hidden))
                                            a.append(str(l1_count))
                                            a.append(str(l2_count))
                                            a.append(str(lr))
                                            a.append(str(dr))
                                            a.append(str(batch_size))
                                            a.append(multi_df['loss'][multi_rows])
                                            a.append(multi_df['val_loss'][multi_rows])
                                            a.append(multi_df['val_mode_accuracy'][multi_rows])
                                            a.append(multi_df['val_mode_crossentropy'][multi_rows])
                                            a.append(multi_df['val_mode_loss'][multi_rows])
                                            a.append(multi_df['mode_accuracy'][multi_rows])
                                            a.append(multi_df['mode_crossentropy'][multi_rows])
                                            a.append(multi_df['mode_loss'][multi_rows])

                                            a.append(multi_df['val_purpose_accuracy'][multi_rows])
                                            a.append(multi_df['val_purpose_crossentropy'][multi_rows])
                                            a.append(multi_df['val_purpose_loss'][multi_rows])
                                            a.append(multi_df['purpose_accuracy'][multi_rows])
                                            a.append(multi_df['purpose_crossentropy'][multi_rows])
                                            a.append(multi_df['purpose_loss'][multi_rows])
                                            multi_file.append(a)

        multi_file = pd.DataFrame(multi_file)
        multi_file.columns = ["mode coeff", "shared hidden layers", "task specific layers", "n_hidden",
                              "l1_count", "l2_count", "learning_rate", "dropout_rate", "batch_size", "total_loss","val_total_loss",
                              "val_mode_acc", "val_mode_ce", "val_mode_loss", "mode_acc", "mode_ce", "mode_loss",
                              "val_purpose_acc", "val_purpose_ce", "val_purpose_loss", "purpose_acc",
                              "purpose_ce", "purpose_loss"]
        multi_file.to_csv("final/multitask.csv", header=True, index=False)

    def stl_result_regular(self):
        for pre_name in self.prediction_item:
            single_file_list = []
            for n_hidden in self.n_hidden_list:
                for lr in self.lr_list:
                    for dr in self.dr_list:
                        for l1_count in self.l1_count_list:
                            for l2_count in self.l2_count_list:
                                for batch_size in self.batch_size_list:
                                    for hidden_layers in self.hidden_layer_list:
                                        single_outputfile = "mean_files/mean_result_" + str(pre_name) + \
                                                            "_" + str(hidden_layers) + "_" + str(n_hidden) + "_" + \
                                                            str(l1_count) + "_" + str(l2_count) + "_" + str(lr) + "_" \
                                                            + str(dr) + "_" + str(batch_size) + ".csv"
                                        single_df = pd.read_csv(single_outputfile)
                                        num_rows = len(single_df) - 1
                                        # print("single epoch: ", num_rows)
                                        a = []
                                        a.append(str(pre_name))
                                        a.append(str(hidden_layers))
                                        a.append(str(n_hidden))
                                        a.append(str(l1_count))
                                        a.append(str(l2_count))
                                        a.append(str(lr))
                                        a.append(str(dr))
                                        a.append(str(batch_size))
                                        a.append(single_df['val_accuracy'][num_rows])
                                        a.append(single_df['val_crossentropy'][num_rows])
                                        a.append(single_df['val_loss'][num_rows])
                                        a.append(single_df['accuracy'][num_rows])
                                        a.append(single_df['crossentropy'][num_rows])
                                        a.append(single_df['loss'][num_rows])
                                        single_file_list.append(a)
            single_file = pd.DataFrame(single_file_list)
            single_file.columns = ["pre_name", "hidden_layers", "n_hidden", "l1_count", "l2_count",
                                   "learning_rate", "dropout_rate", "batch_size",
                                   "single_test_acc", "single_test_ce", "single_test_loss",
                                   "single_train_acc", "single_train_ce", "single_train_loss"]
            single_file.to_csv("final/" + pre_name + '.csv', header=True, index=False)

    def mtl_stl_result_regular(self):
        m = pd.read_csv("final/mode.csv")
        p = pd.read_csv("final/purpose.csv")
        mtl = pd.read_csv("final/multitask.csv")

        final_list = []
        col_name = ['mode coeff', "shared hidden layers", "task specific layers", 'hidden_layers',
                    'n_hidden', 'l1_count', 'l2_count', 'learning_rate', 'dropout_rate', 'batch_size',
                    "total_loss", "val_total_loss"]

        col_name_2 = ["single_test_acc", "single_test_ce", "single_test_loss", "single_train_acc",
                      "single_train_ce","single_train_loss"]
        mtl["hidden_layers"] = mtl['shared hidden layers'] + mtl['task specific layers']
        # print(len(m["hidden_layers"]))
        for i in range(len(mtl["hidden_layers"])):
            for j in range(len(m["hidden_layers"])):
                copy_list_m = []
                if str(mtl['hidden_layers'][i]) == str(m['hidden_layers'][j]) and str(mtl['n_hidden'][i]) == str(
                        m['n_hidden'][j]) and \
                        str(mtl['l1_count'][i]) == str(m['l1_count'][j]) and str(mtl['l2_count'][i]) == str(
                    m['l2_count'][j]) and \
                        str(mtl['learning_rate'][i]) == str(m['learning_rate'][j]) and str(
                    mtl['dropout_rate'][i]) == str(m['dropout_rate'][j]) \
                        and str(mtl['batch_size'][i]) == str(m['batch_size'][j]):
                    # print("true")
                    for col in col_name:
                        copy_list_m.append(mtl[col][i])
                    for col_2 in col_name_2:
                        copy_list_m.append(m[col_2][j])
                    copy_list_m.insert(12, mtl["val_mode_acc"][i])
                    copy_list_m.insert(14, mtl["val_mode_ce"][i])
                    copy_list_m.insert(16, mtl["val_mode_loss"][i])
                    copy_list_m.insert(18, mtl["mode_acc"][i])
                    copy_list_m.insert(20, mtl["mode_ce"][i])
                    copy_list_m.insert(22, mtl["mode_loss"][i])
                    # final_list.append(copy_list_m)
                copy_list_p = []
                if str(mtl['hidden_layers'][i]) == str(p['hidden_layers'][j]) and str(mtl['n_hidden'][i]) == str(
                        p['n_hidden'][j]) and \
                        str(mtl['l1_count'][i]) == str(p['l1_count'][j]) and str(mtl['l2_count'][i]) == str(
                    p['l2_count'][j]) and \
                        str(mtl['learning_rate'][i]) == str(p['learning_rate'][j]) and str(
                    mtl['dropout_rate'][i]) == str(p['dropout_rate'][j]) and \
                        str(mtl['batch_size'][i]) == str(p['batch_size'][j]):
                    # print("P true")
                    for col_2 in col_name_2:
                        copy_list_p.append(p[col_2][j])
                    copy_list_p.insert(0, mtl["val_purpose_acc"][i])
                    copy_list_p.insert(2, mtl["val_purpose_ce"][i])
                    copy_list_p.insert(4, mtl["val_purpose_loss"][i])
                    copy_list_p.insert(6, mtl["purpose_acc"][i])
                    copy_list_p.insert(8, mtl["purpose_ce"][i])
                    copy_list_p.insert(10, mtl["purpose_loss"][i])
                copy_list = copy_list_m + copy_list_p
                if len(copy_list) > 0:
                    final_list.append(copy_list)
        final_list = pd.DataFrame(final_list)
        column_m = ["multi_test_mode_acc", "single_test_mode_acc", "multi_test_mode_ce",
                    "single_test_mode_ce", "multi_test_mode_loss", "single_test_mode_loss",
                    "multi_train_mode_acc", "single_train_mode_acc", "multi_train_mode_ce",
                    "single_train_mode_ce", "multi_train_mode_loss", "single_train_mode_loss"]
        column_p = ["multi_test_purpose_acc", "single_test_purpose_acc", "multi_test_purpose_ce",
                    "single_test_purpose_ce", "multi_test_purpose_loss", "single_test_purpose_loss",
                    "multi_train_purpose_acc", "single_train_purpose_acc", "multi_train_purpose_ce",
                    "single_train_purpose_ce", "multi_train_purpose_loss", "single_train_purpose_loss"]
        final_list.columns = col_name + column_m + column_p

        file_name = "final/result.csv"
        final_list.to_csv(file_name, index=False)