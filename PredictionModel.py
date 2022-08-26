import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from datetime import datetime
from LossHistory import LossHistory
import keras.backend as K
from numpy.random import permutation
from keras import regularizers


class PredictionModel:
    """model of multi-task and single-task prediction"""

    def __init__(self, X, input_size, Y_m, Y_p, epoch, iterations, coe_list, single_pre_item_list,
                 n_hidden_list, lr_list, dr_list, l1_count_list, l2_count_list, batch_size_list,activation_name, SAVE_MODEL):
        '''
        X, input_size, Y_m, Y_p, epoch, iterations, coe_list, single_pre_item,

        '''
        self.X = X
        self.input_size = input_size
        self.Y_m = Y_m
        self.Y_p = Y_p
        self.epoch = epoch
        self.iterations = iterations
        self.coe_list = coe_list
        self.prediction_item = single_pre_item_list
        self.num_class_mode = 6
        self.num_class_purpose = 7
        self.SAVE_MODEL = SAVE_MODEL

        # parameters of model
        self.n_hidden_list = n_hidden_list
        self.lr_list = lr_list  # 1e-3
        self.dr_list = dr_list  # 1e-2
        self.l1_count_list = l1_count_list  # 1e-5
        self.l2_count_list = l2_count_list  # 1e-3
        self.batch_size_list = batch_size_list  # 200
        self.activation_name = activation_name

        # 'relu' or 'tanh' or 'sigmoid' or ...
        # keras.layers.advanced_activations.PReLU(alpha_initializer='zeros',alpha_regularizer=None,alpha_constraint=None,shared_axes=None)
        # keras.layers.advanced_activations.LeakyReLU(alpha=0.3)

    def multitask_prediction(self, shared_hidden_layers, task_specific_layers):
        """Multitask prediction model using hard parameter sharing"""
        print("Multitask prediction using hard parameter sharing is running...")
        self.shared_hidden_layers = shared_hidden_layers
        self.task_specific_layers = task_specific_layers

        for Iter in range(self.iterations):  # iterations
            for mode_coefficient in self.coe_list:
                for n_hidden in self.n_hidden_list:
                    for lr in self.lr_list:
                        for dr in self.dr_list:
                            for l1_count in self.l1_count_list:
                                for l2_count in self.l2_count_list:
                                    for batch_size in self.batch_size_list:
                                        csvname = "results/result_" + str(mode_coefficient) + "_" + \
                                                  str(self.shared_hidden_layers) + "_" + str(self.task_specific_layers) + "_" + str(n_hidden) + "_" + \
                                                  str(l1_count) + "_"+ str(l2_count) + "_"+ str(lr) + "_"+ str(dr) + "_"+ str(batch_size) + "_" + str(Iter) + ".csv"
                                        # file path and name  # file path and name

                                        time_before = datetime.now()
                                        print("程序执行之前，当前时间：" + str(time_before))

                                        # input layer
                                        features_input = keras.Input(
                                            shape=(self.input_size,), name="Features",
                                        )
                                        # shared hidden layers
                                        hidden = Dense(n_hidden, activation=self.activation_name,
                                                       kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(features_input)
                                        hidden = Dropout(dr)(hidden)
                                        if self.shared_hidden_layers > 1:
                                            for i in range(self.shared_hidden_layers - 1):
                                                hidden = Dense(n_hidden, activation=self.activation_name,kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(hidden)
                                                hidden = Dropout(dr)(hidden)
                                        # the first mode task specific layer
                                        mode_pred = Dense(n_hidden, activation=self.activation_name,kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(hidden)
                                        mode_pred = Dropout(dr)(mode_pred)
                                        # the first purpose task specific layer
                                        purpose_pred = Dense(n_hidden, activation=self.activation_name,
                                                             kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(hidden)
                                        purpose_pred = Dropout(dr)(purpose_pred)

                                        if self.task_specific_layers > 1:
                                            for j in range(self.task_specific_layers - 1):
                                                # add mode layers
                                                mode_pred = Dense(n_hidden, activation=self.activation_name, kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(mode_pred)
                                                mode_pred = Dropout(dr)(mode_pred)
                                                # add purpose layers
                                                purpose_pred = Dense(n_hidden, activation=self.activation_name, kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count))(
                                                    purpose_pred)
                                                purpose_pred = Dropout(dr)(purpose_pred)
                                        # last mode layers
                                        mode_pred = Dense(self.num_class_mode, activation='softmax', name="mode")(mode_pred)
                                        # last purpose layers
                                        purpose_pred = Dense(self.num_class_purpose, activation='softmax', name="purpose")(purpose_pred)

                                        # Instantiate an end-to-end model predicting both mode and purpose
                                        model = keras.Model(
                                            inputs=[features_input],
                                            outputs=[mode_pred, purpose_pred],
                                        )

                                        model.compile(optimizer='adam',
                                                      loss={
                                                          "mode": keras.losses.categorical_crossentropy,
                                                          "purpose":keras.losses.categorical_crossentropy,
                                                      },
                                                      loss_weights={"mode": mode_coefficient, "purpose": 1 - mode_coefficient},
                                                      metrics=['crossentropy', 'accuracy']
                                                      )

                                        call_backlist = LossHistory()  # LossHistory
                                        call_backlist.get_csv_name(csvname)  # one name per iteration

                                        # shuffle data before fitting
                                        perm = permutation(self.X.index)
                                        X = self.X.reindex(perm)
                                        Y_m = self.Y_m.reindex(perm)
                                        Y_p = self.Y_p.reindex(perm)

                                        K.set_value(model.optimizer.lr, lr)
                                        model.fit(
                                            {"Features": X},
                                            {"mode": Y_m, "purpose": Y_p},
                                            epochs=self.epoch,
                                            batch_size=batch_size,
                                            validation_split=0.3,  # train-test split
                                            callbacks=[call_backlist]  # accuracy of each epoch
                                        )

                                        # add two columns ["epoch","Iter"] in the csv file

                                        l1 = []
                                        l2 = []
                                        for i in range(self.epoch):
                                            l2.append(Iter)
                                            l1.append(i)
                                        df = pd.read_csv(csvname)
                                        df['epoch'] = l1
                                        df['iteration'] = l2
                                        df.to_csv(csvname,index="epoch")

                                        # parameters of plotting
                                        # keras.utils.plot_model(model, "plot/multitask_model" + "_" + str(self.shared_hidden_layers) + "_" + str(self.task_specific_layers) + ".png", show_shapes=False)
                                        # model.summary()

                                        # 模型运行时间
                                        time_after = datetime.now()
                                        print("程序执行之前，当前时间：" + str(time_after))
                                        print("程序执行时间：" + str(time_after - time_before))

                                        if self.SAVE_MODEL is True:
                                            model.save("model/MTL_M_model_" + str(mode_coefficient) + "_" + str(Iter) + ".h5")



    def singletask_prediction(self, hidden_layers):
        """Multitask prediction model"""
        print("Singletask prediction is running...")

        self.hidden_layers = hidden_layers
        for pre_name in self.prediction_item:
            for n_hidden in self.n_hidden_list:
                for lr in self.lr_list:
                    for dr in self.dr_list:
                        for l1_count in self.l1_count_list:
                            for l2_count in self.l2_count_list:
                                for batch_size in self.batch_size_list:
                                    num_class = self.num_class_mode if pre_name == "mode" else self.num_class_purpose  # output dimensions mode=4 && purpose=5
                                    for Iter in range(self.iterations):
                                        # csv name of each iteration
                                        csvname = "results/result_" + str(pre_name) + "_" + \
                                                  str(self.hidden_layers) + "_" + str(n_hidden) + "_" + \
                                                  str(l1_count) + "_" + str(l2_count) + "_" + str(lr) + "_" + str(dr) + "_" + str(batch_size) +\
                                                  "_" + str(Iter) + ".csv"

                                        time_before = datetime.now()
                                        print("程序执行之前，当前时间：" + str(time_before))

                                        model = Sequential()
                                        model.add(Dense(n_hidden, input_shape=(self.input_size,),
                                                        kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count)))
                                        model.add(Dropout(dr))
                                        for i in range(hidden_layers-1):
                                            model.add(Dense(n_hidden, activation=self.activation_name, kernel_regularizer=regularizers.l1_l2(l1=l1_count, l2=l2_count)))
                                            model.add(Dropout(dr))

                                        model.add(Dense(num_class, activation='softmax'))

                                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['crossentropy', 'accuracy'])

                                        call_backlist = LossHistory()
                                        call_backlist.get_csv_name(csvname)

                                        # shuffle data before fitting
                                        perm = permutation(self.X.index)
                                        X = self.X.reindex(perm)
                                        Y_m = self.Y_m.reindex(perm)
                                        Y_p = self.Y_p.reindex(perm)

                                        # print(X)
                                        # print(Y_m)
                                        # print(Y_p)
                                        Y = Y_m if pre_name == "mode" else Y_p  # which task? mode or purpose

                                        K.set_value(model.optimizer.lr, lr)
                                        model.fit(X, Y, epochs=self.epoch, batch_size=batch_size, validation_split=0.3,
                                                  callbacks=[call_backlist])

                                        l1 = []
                                        l2 = []
                                        for i in range(self.epoch):
                                            l2.append(Iter)
                                            l1.append(i)
                                        df = pd.read_csv(csvname)
                                        df['epoch'] = l1
                                        df['iteration'] = l2
                                        df.to_csv(csvname,index="epoch")

                                        # parameters of plotting
                                        # keras.utils.plot_model(model, "plot/" + str(pre_name) + "_" + str(hidden_layers) + "_model.png", show_shapes=False)
                                        # model.summary()

                                        # 模型运行时间
                                        time_after = datetime.now()
                                        print("程序执行之前，当前时间：" + str(time_after))
                                        print("程序执行时间：" + str(time_after - time_before))

                                        if self.SAVE_MODEL is True:
                                            if pre_name == "mode":
                                                model.save("model/STL_M_model_" + str(Iter) + ".h5")
                                            else:
                                                model.save("model/STL_P_model_" + str(Iter) + ".h5")