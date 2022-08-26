import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from keras.models import load_model
import re

from Tools import Tools
tools = Tools()

class ResultAnalysisAndPlot:

    def __init__(self, X, X_train_raw, all_variables, cont_variables, iterations, mode_coe,
                 PLOT_SUBSTITUTE_PATTERN, PLOT_CHOICE_PROBABILITY, PLOT_PROBABILITY_DERIVATIVE):

        self.X = X
        self.X_train_raw = np.array(X_train_raw)
        self.all_variables = all_variables
        self.cont_variables = cont_variables
        self.iterations = iterations
        self.mode_coe = mode_coe
        self.PLOT_SUBSTITUTE_PATTERN = PLOT_SUBSTITUTE_PATTERN
        self.PLOT_CHOICE_PROBABILITY = PLOT_CHOICE_PROBABILITY
        self.PLOT_PROBABILITY_DERIVATIVE = PLOT_PROBABILITY_DERIVATIVE

        # 坐标轴字体
        self.fontname = r"c:/windows/fonts/simsun.ttc"
        self.font = FontProperties(fname=self.fontname, size=16)
        self.font_mini = FontProperties(fname=self.fontname, size=12)
        # 通用字体
        self.fontsize = 16
        self.fontsize_mini = 12

        self.modes_or_purposes = [] # 根据预测内容等于 self.modes 或 self.purposes

        self.modes = ['步行','公共汽车','地铁轻轨','出租车或网约车','私家车','定制公交或共享汽车']
        self.purposes = ['上班','上学','回家','购物','娱乐','旅游或出差','接送家人']

        self.plot_vars_idx = [[2], [3], [4]]
        self.plot_vars_standard_idx = [[2], [3], [4]]
        self.plot_var_names = [['出行时间 (分钟)'], ['出行距离 (公里)'], ['出行费用 (元)']]

        self.choice_pro_plot_m_or_p = []    # 根据预测内容等于 self.choice_pro_plot_modes 或 self.choice_pro_plot_purposes
        self.choice_pro_plot_modes = [
                                      ["步行"],
                                      ["公共汽车"],
                                      ["地铁轻轨"],
                                      ["出租车或网约车"],
                                      ["私家车"],
                                      ["定制公交或共享汽车"]
                                     ]

        self.choice_pro_plot_purposes = [
                                      ["上班"],
                                      ["上学"],
                                      ["回家"],
                                      ["购物"],
                                      ["娱乐"],
                                      ["旅游或出差"],
                                      ["接送家人"]
                                     ]

    def forward(self,model_type):
        """
         compute single-task probability and probability derivative
         plot single-task substitute_pattern, choice probability and probability derivative
        """

        if  model_type == "MTL_M":
            model_type_cn = "多任务预测(出行方式)"
            model_name_start = "MTL_M_model_" + str(self.mode_coe) + "_"
            self.num_class_m_or_p = 6
            self.modes_or_purposes = self.modes
            self.choice_pro_plot_m_or_p = self.choice_pro_plot_modes
        elif model_type == "MTL_P":
            model_type_cn = "多任务预测(出行目的)"
            model_name_start = "MTL_M_model_" + str(self.mode_coe) + "_"    # MTL_M 与 MTL_P 是一个模型, 保存为了 MTL_M...
            self.num_class_m_or_p = 7
            self.modes_or_purposes = self.purposes
            self.choice_pro_plot_m_or_p = self.choice_pro_plot_purposes

        elif model_type == "STL_M":
            model_type_cn = "单任务预测（出行方式）"
            model_name_start = model_type + "_model_"
            self.num_class_m_or_p = 6
            self.modes_or_purposes = self.modes
            self.choice_pro_plot_m_or_p = self.choice_pro_plot_modes
        elif model_type == "STL_P":
            model_type_cn = "单任务预测（出行目的）"
            model_name_start = model_type + "_model_"
            self.num_class_m_or_p = 7
            self.modes_or_purposes = self.purposes
            self.choice_pro_plot_m_or_p = self.choice_pro_plot_purposes
        else:
            return print("输入有误,forward()可以接收的参数有：MTL_M、STL_M、MTL_P、STL_P，请核对")
        print("forward(" + model_type + ") 正在执行 ...")

        # 初始化所需矩阵
        self.choice_prob = np.zeros((len(self.cont_variables), self.iterations, self.num_class_m_or_p, len(self.X)))
        self.prob_derivative = np.zeros((len(self.cont_variables), self.iterations, self.num_class_m_or_p, len(self.X)))
        self.average_cp = np.zeros((len(self.cont_variables), self.iterations, self.num_class_m_or_p, len(self.X)))
        self.average_pd = np.zeros((len(self.cont_variables), self.iterations, self.num_class_m_or_p, len(self.X)))

        for Iter in range(self.iterations):
            # load model
            model = load_model("model/" + model_name_start + str(Iter) + ".h5")
            # Computing the probability and probability derivative
            choice_prob, prob_derivative = self.compute_prob_curve(model_type, self.X, self.cont_variables, model)
            # add "prob_derivative" of each iteration to "self.prob_derivative"
            self.prob_derivative[:, Iter, :, :] = np.array(prob_derivative)
            # add "choice_prob" of each iteration to "self.choice_prob"
            self.choice_prob[:, Iter, :, :] = np.array(choice_prob)
        self.average_cp = np.mean(self.choice_prob, axis=1)
        self.average_pd = np.mean(self.prob_derivative, axis=1)

        for plot_vars_idx, plot_vars_standard_idx, plot_var_names in zip(
                self.plot_vars_idx, self.plot_vars_standard_idx, self.plot_var_names):

            if self.PLOT_SUBSTITUTE_PATTERN is True:
                # 绘制各种交通方式或出行目的随着某一变量（出行时间、出行距离、出行费用）增长的变化
                self.plot_substitute_pattern(model_type_cn, plot_vars_idx, plot_vars_standard_idx, plot_var_names)

            for plot_m_or_p in self.choice_pro_plot_m_or_p:
                if self.PLOT_CHOICE_PROBABILITY is True:
                    # 绘制单一交通方式或出行目的随着某一变量（出行时间、出行距离、出行费用）增长的变化
                    self.plot_choice_prob(model_type_cn, plot_m_or_p, plot_vars_idx, plot_vars_standard_idx, plot_var_names)
                if self.PLOT_PROBABILITY_DERIVATIVE is True:
                    # 绘制单一交通方式或出行目的对某一变量（出行时间、出行距离、出行费用）的波动敏感性
                    self.plot_prob_derivative(model_type_cn, plot_m_or_p, plot_vars_idx, plot_vars_standard_idx, plot_var_names)

        print("forward(" + model_type + ") 执行完成 ...")

    def compute_prob_curve(self, model_type, input_x, cont_vars, loaded_model):
        """Computing the probability and probability derivative using a market average person
        Args:
            input_x (pd.DataFrame): data frame of X_train (processed)
            cont_vars (list of str): names of cont_variables
            loaded_model (classifier model): a model of multi-task DNN classifier
        """
        input_x = np.array(input_x)
        x_avg = np.mean(input_x, axis=0)
        x_feed = np.repeat(x_avg, len(input_x)).reshape(np.size(input_x, axis=1), len(input_x)).T
        choice_prob = []
        prob_derivative = []
        for var in cont_vars:
            # get the idx of var in all_variables
            idx = self.all_variables.index(var)
            x_feed[:, idx] = input_x[:, idx]
            # compute probability
            if model_type == "MTL_M":
                temp_prob, temp_prob_p = loaded_model.predict(x_feed)
            elif model_type == "MTL_P":
                temp_prob_m, temp_prob = loaded_model.predict(x_feed)
            else:
                temp_prob = loaded_model.predict(x_feed)
            choice_prob.append(np.array(temp_prob).T)

            # compute gradient
            # deciding the interval. Should be greater than zero.
            # If the interval is too small, the probability change is insignificant.
            # Tree-based models are insensitive to very small changes in feature values
            increment = np.ptp(x_feed[:, idx]) / 10
            if increment <= 0:
                increment = 1e-3
            # increase x_feed by increment
            x_feed[:, idx] = x_feed[:, idx] + increment
            if model_type == "MTL_M":
                new_prob_m, new_prob_p = loaded_model.predict(x_feed)
                gradient = (new_prob_m - temp_prob) / increment
            elif model_type == "MTL_P":
                new_prob_m, new_prob_p = loaded_model.predict(x_feed)
                gradient = (new_prob_p - temp_prob) / increment
            else:
                new_prob = loaded_model.predict(x_feed)
                gradient = (new_prob - temp_prob) / increment
            prob_derivative.append(np.array(gradient).T)
            # restore x_feed to x_avg
            x_feed[:, idx] = x_avg[idx]
        return np.array(choice_prob), np.array(prob_derivative)

    def plot_substitute_pattern(self, model_type_cn, plot_vars_idx, plot_vars_standard_idx, plot_var_names):
        """ plot travel modes or purposes substitute_pattern

        Args:
            plot_vars_idx (list of int): containing the index of plot_vars in all_vars
            plot_vars_standard_idx (list of int): containing the index of plot_vars in standard_vars
            plot_var_names (list of str): containing labels on the x axis
        """

        colors = tools.generate_colors()
        for name, v, vs in zip(plot_var_names, plot_vars_idx, plot_vars_standard_idx):
            fig, ax = plt.subplots(figsize=(8, 8))
            # plt.rcParams.update({'font.size': self.fontsize_mini})
            for j in range(self.num_class_m_or_p):
                df = np.insert(self.choice_prob[vs, :, j, :], 0, self.X_train_raw[:, v], axis=0)
                df = df[:, df[0, :].argsort()]

                # plot the line of each iteration
                # for Iter in range(self.iterations):
                #     ax.plot(df[0, :], df[Iter + 1, :], linewidth=1, alpha=0.5, color=colors[j], label='')

            for j in range(self.num_class_m_or_p):
                plot = sorted(zip(self.X_train_raw[:, v], np.mean(self.choice_prob[vs, :, j, :], axis=0)))
                ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color=colors[j], label=self.modes_or_purposes[j])

            xlabels = np.linspace(0, np.percentile(self.X_train_raw[:, v], 95), 11)
            ylabels = np.linspace(0, 1, 11)

            list_match = re.search("(.*) (.*)", plot_var_names[0])
            plot_var = list_match[1]

            ax.set_xticks(xlabels)
            ax.set_yticks(ylabels)
            ax.set_xlim([0, np.percentile(self.X_train_raw[:, v], 95)])
            ax.set_ylim([-0.05, 1.0])
            label_format = '{:,.1f}'

            ax.set_xticklabels([label_format.format(x) for x in xlabels], font=self.font_mini)
            ax.set_yticklabels([label_format.format(y) for y in ylabels], font=self.font_mini)
            if plot_var == "出行费用":
                # label_format = '{:,.1f}'
                ax.set_xticklabels(['{:,.0f}'.format(x) for x in xlabels], font=self.font_mini)

            ax.set_ylabel("选择概率", font=self.font)
            ax.set_xlabel(name, font = self.font)

            # legend = ax.legend(loc='lower right') # upper right
            legend = ax.legend()
            for text in legend.texts:
                text.set_font_properties(self.font_mini)

            substitute_name = model_type_cn + "_" + str(plot_var) + "_选择概率" + ".png"
            plt.savefig("plot/选择概率/" + substitute_name, dpi=300)
            plt.close()

    def plot_choice_prob(self,model_type_cn, plot_m_or_p, plot_vars_idx, plot_vars_standard_idx, plot_var_names, highlight=[], highlight_label=[]):
        """Plot the choice probabilities of any one mode or purpose with varying Trip distance or Trip time.
            Note that plot_m_or_p and plot_vars should be paired and have the same length.

        Args:
            plot_m_or_p (list of str): containing modes or purposes.
            plot_vars_idx (list of int): containing the index of plot_vars in all_variables
            plot_vars_standard_idx (list of int): containing the index of plot_vars in cont_variables
            plot_var_names (list of str): containing labels on the x axis, e.g. 'Trip_distance (miles)','Trip_time (minutes)'
            highlight (list, optional): highlight the line of any one iterations. Defaults to [].
            highlight_label (list, optional): containing labels of highlight iterations. Defaults to [].
        """
        colors = tools.generate_colors()
        plt.rcParams.update({'font.size': self.fontsize_mini})
        for i, v, vs in zip([x for x in range(len(plot_vars_idx))], plot_vars_idx, plot_vars_standard_idx):
            fig, ax = plt.subplots(figsize=(8, 8))

            # mode or purpose for plotting
            m_or_p = plot_m_or_p[i]
            j = self.modes_or_purposes.index(m_or_p)
            list_match = re.search("(.*) (.*)", plot_var_names[0])
            plot_var = list_match[1]

            # plot the line of each iteration
            for Iter in range(self.iterations):
                plot = sorted(zip(self.X_train_raw[:, v], self.choice_prob[vs, Iter, j, :]))
                if Iter not in highlight:
                    ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1, color='silver', label='')
            for Iter in highlight:
                plot = sorted(zip(self.X_train_raw[:, v], self.choice_prob[vs, Iter, j, :]))
                ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1,
                                   color=colors[highlight.index(Iter)], label=highlight_label[highlight.index(Iter)])
            df = pd.DataFrame(np.array([self.X_train_raw[:, v], self.average_cp[vs, j, :]]).T, columns=['var', 'cp']).groupby('var', as_index=False).mean()[['var', 'cp']]
            df.sort_values(by='var', inplace=True)
            ax.plot(df['var'], df['cp'], linewidth=3, color='k', label='Average')

            xlabels = np.linspace(0, np.percentile(self.X_train_raw[:, v], 95), 11)
            ylabels = np.linspace(0, 1, 11)
            m_or_p = plot_m_or_p[i]
            ax.set_ylabel(m_or_p, font=self.font)
            ax.set_xlabel(plot_var_names[i], font=self.font)
            ax.set_xticks(xlabels)
            ax.set_yticks(ylabels)
            ax.set_xlim([0, np.percentile(self.X_train_raw[:, v], 95)])
            ax.set_ylim([-0.05, 1.05])
            ax.legend(fancybox=True, framealpha=0.5)
            label_format = '{:,.1f}'
            ax.set_yticklabels([label_format.format(y) for y in ylabels], font=self.font_mini)
            ax.set_xticklabels([label_format.format(x) for x in xlabels], font=self.font_mini)
            if plot_var == "出行费用":
                ax.set_xticklabels(['{:,.0f}'.format(x) for x in xlabels], font=self.font_mini)

            legend = ax.legend(fancybox=True, framealpha=0.5)
            for text in legend.texts:
                text.set_font_properties(self.font_mini)

            choice_prob_name = model_type_cn + "_" + str(m_or_p) + "_选择概率_"+ str(plot_var) + ".png"
            plt.savefig("plot/选择概率拆分/" + choice_prob_name, dpi=300)
            plt.close()

    def plot_prob_derivative(self,model_type_cn, plot_m_or_p, plot_vars_idx, plot_vars_standard_idx, plot_var_names, highlight=[], highlightlabel=[]):
        """Plot probability derivative. mkt_prob_derivative is used here.
        Note that plot_m_or_p and plot_vars should be paired and have the same length.

        Args:
            plot_m_or_p (list of str): containing modes or purposes
            plot_vars_idx (list of int): containing the index of plot_vars in all_vars
            plot_vars_standard_idx (list of int): containing the index of plot_vars in standard_vars
            plot_var_names (list of str): containing labels on the x axis, e.g. 'drive_cost ($)'
            highlight (list of int, optional): list containing the index of highlighted models. Defaults to [].
            highlightlabel (list of str, optional): list containing the index of highlighted models. Defaults to [].
        """

        colors = tools.generate_colors()
        plt.rcParams.update({'font.size': self.fontsize_mini})

        for i, v, vs in zip([x for x in range(len(plot_vars_idx))], plot_vars_idx, plot_vars_standard_idx):
            fig, ax = plt.subplots(figsize=(5, 5))

            m_or_p = plot_m_or_p[i]
            j = self.modes_or_purposes.index(m_or_p)

            list_match = re.search("(.*) (.*)", plot_var_names[0])
            plot_var = list_match[1]

            for index in range(self.iterations):
                df = pd.DataFrame(np.array([self.X_train_raw[:, v], self.prob_derivative[vs, index, j, :]]).T,
                                  columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
                df.sort_values(by='var', inplace=True)
                if index not in highlight:
                    ax.plot(df['var'], df['pd'], linewidth=2, color='silver', label='')
            for index in highlight:
                df = pd.DataFrame(np.array([self.X_train_raw[:, v], self.prob_derivative[vs, index, j, :]]).T,
                                columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
                df.sort_values(by='var', inplace=True)
                ax.plot(df['var'], df['pd'], linewidth=2, color=colors[highlight.index(index)],
                                   label=highlightlabel[highlight.index(index)])

            df = pd.DataFrame(np.array([self.X_train_raw[:, v], self.average_pd[vs, j, :]]).T, columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
            df.sort_values(by='var', inplace=True)
            ax.plot(df['var'], df['pd'], linewidth=3, color='k',label='Average')


            m_or_p = plot_m_or_p[i]
            ax.set_ylabel(m_or_p, font=self.font)
            ax.set_xlabel(plot_var_names[i], font=self.font)

            xlabels = np.linspace(0, np.percentile(self.X_train_raw[:, v], 95), 11)
            ax.set_xticks(xlabels)
            ax.set_xlim([0, np.percentile(self.X_train_raw[:, v], 95)])
            ax.set_xticklabels(['{:,.1f}'.format(x) for x in xlabels], font=self.font_mini)
            if plot_var == "出行费用":
                ax.set_xticklabels(['{:,.0f}'.format(x) for x in xlabels], font=self.font_mini)

            ylabels = np.linspace(-0.10, 0.10, 11)
            ax.set_yticks(ylabels)
            ax.set_ylim([-0.11, 0.11])
            ax.set_yticklabels(['{:,.3f}'.format(y) for y in ylabels], font=self.font_mini)

            # legend = ax.legend(fancybox=True, framealpha=0.5)
            # for text in legend.texts:
            #     text.set_font_properties(self.font_mini)

            prob_derivative_name = model_type_cn + "_" + str(m_or_p) + "_选择稳定性_" + str(plot_var) + ".png"
            plt.savefig("plot/选择概率稳定性/" + prob_derivative_name, dpi=300)
            plt.close()