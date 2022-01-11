import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time

class MyDataset:
    '''The present class is used to load, analyze and preprocess a
    dataset. The original dataset is saved in "self.original_data",
    while the manipulated one is saved in "self.__data".
    Insert "filename" and "dataset_path" to initialize the class.'''
    def __init__(self, dataset_path, filename):
        self.dataset_path = dataset_path
        self.filename = filename

    def load_csv(self, skiprows=0, delimiter=';', flag_return_data=False, class_name=None):
        data = pd.read_csv(os.path.join(self.dataset_path, self.filename), \
            skiprows = skiprows, delimiter = delimiter)
        self.__data = data
        self.__header =  self.__data.columns
        self.__data_shape = self.__data.shape
        print(self.filename+' dataset was loaded!')
        print(self.__data_shape)
        if flag_return_data:
            return self.__data
        if class_name==None:
            self.__class_name = self.__header[-1]
        else:
            self.__class_name = class_name

    def load_txt(self, skiprows=0, delimiter=',', flag_return_data=False, class_name=None):
        data = np.loadtxt(os.path.join(self.dataset_path, self.filename), \
            skiprows = skiprows, delimiter = delimiter)
        self.__header = np.arange(data.shape[1])
        data = pd.DataFrame(data, columns=self.__header)
        self.__data = data
        self.__data_shape = self.__data.shape
        print(self.filename+' dataset was loaded!')
        print(self.__data_shape)
        if flag_return_data:
            return self.__data
        if class_name==None:
            self.__class_name = self.__header[-1]
        else:
            self.__class_name = class_name

    def change_header(self, header, flag_return_data=False):
        self.__header = header
        self.__data.columns = self.__header
        if flag_return_data:
            return self.__data

    def get_data(self):
        return self.__data

    def separate_data_and_target(self, target_columns, flag_return_data=True):
        self.__target_columns_names = self.__data.columns[target_columns]
        self.__target = self.__data[self.__target_columns_names]
        self.__data = self.__data.drop(columns=self.__target_columns_names)
        self.__header = self.__data.columns
        self.__data_shape = self.__data.shape
        print('Input data:', self.__data_shape, '- Output class:', \
            self.__target.shape)
        if flag_return_data:
            return self.__data, self.__target

    def check_nan(self, flag_plots=False):
        print('Number of NaN values')
        print(pd.DataFrame(self.__data.isna().sum(), \
            columns = ['NaN Values']).T)
        self.__isnan = self.__data.isna().astype('int')
        self.__features_number = len(self.__data.columns)
        self.__plot_grid_number = int(np.ceil(np.sqrt(
            self.__features_number)))
        if flag_plots:
            figure, ax = plt.subplots(nrows=self.__plot_grid_number, \
                ncols=self.__plot_grid_number, figsize=(10,10))
            ax = ax.flatten()
            for feat in range(self.__features_number):
                ax[feat].plot(self.__isnan[self.__isnan.columns[feat]])
                ax[feat].set_title(self.__isnan.columns[feat])
                plt.tight_layout()

    def check_null(self, flag_plots=False):
        print('Number of Null values')
        print(pd.DataFrame(self.__data.isnull().sum(), \
            columns = ['Null Values']).T)
        self.__isnull = self.__data.isnull().astype('int')
        self.__features_number = len(self.__data.columns)
        self.__plot_grid_number = int(np.ceil(np.sqrt(
            self.__features_number)))
        if flag_plots:
            figure, ax = plt.subplots(nrows=self.__plot_grid_number, \
                ncols=self.__plot_grid_number, figsize=(10,10))
            ax = ax.flatten()
            for feat in range(self.__features_number):
                ax[feat].plot(self.__isnull[self.__isnan.columns[feat]])
                ax[feat].set_title(self.__isnan.columns[feat])
                plt.tight_layout()

    def check_outliers(self, flag_plots=False, flag_print_output=True):
        ## Checking which variables are continuous
        MyDataset.get_float_attributes(self)
        ## Interquartile Rule
        MyDataset.interquartile_rule(self)

        output_dataframe = pd.DataFrame([self.__quartile_1st, \
            self.__median, self.__quartile_3rd, self.__under_lower_bound, \
                self.__above_upper_bound, self.__total_outliers, \
                    self.__outliers_percentual], columns = \
                self.__float_columns, index = ['1st Quartile', \
                    'Median', '3rd Quartile', 'Under Lower Bound', \
                        'Above Upper Bound', 'Total Outliers (abs)', \
                            'Total Outliers (%)'])
        if flag_print_output:
            print(output_dataframe)

        ## Boxplots
        if flag_plots:
            figure, ax = plt.subplots(nrows=self.__plot_grid_number, \
                ncols=self.__plot_grid_number, figsize=(10,10))
            ax = ax.flatten()
            for feat in range(self.__float_features_number):
                ax[feat].boxplot(
                    self.__float_data[self.__float_columns[feat]])
                ax[feat].set_title(self.__float_columns[feat])
                plt.tight_layout()

    def get_float_attributes(self):
        self.__dtypes = self.__data.dtypes
        self.__float_features_number = (self.__dtypes=='float').sum()
        self.__float_columns = self.__data.columns[self.__dtypes=='float']
        self.__float_data = self.__data[self.__float_columns]
        self.__plot_grid_number = int(np.ceil(np.sqrt(
            self.__float_features_number)))

    def interquartile_rule(self):
        self.__quartile_1st = np.quantile(self.__float_data, 0.25, axis = 0)
        self.__median = np.median(self.__float_data, axis = 0)
        self.__quartile_3rd = np.quantile(self.__float_data, 0.75, axis = 0)
        self.__interquartile = self.__quartile_3rd - self.__quartile_1st
        self.__upper_bound = self.__quartile_3rd + (1.5 * self.__interquartile)
        self.__lower_bound = self.__quartile_1st - (1.5 * self.__interquartile)
        self.__under_lower_bound = (self.__float_data < \
            self.__lower_bound).sum(axis = 0)
        self.__above_upper_bound = (self.__float_data > \
            self.__upper_bound).sum(axis = 0)
        self.__total_outliers = self.__under_lower_bound + \
            self.__above_upper_bound
        self.__outliers_percentual = self.__total_outliers/ \
            self.__data_shape[0]
    
    def remove_nan(self):
        self.__data = self.__data.dropna()
        return self.__data

    def remove_outliers(self, flag_print_check_outliers=False):
        MyDataset.check_outliers(self, flag_print_output = \
            flag_print_check_outliers)
        self.__under_lower_bound_boolean = (self.__float_data < \
            self.__lower_bound)
        self.__above_upper_bound_boolean = (self.__float_data > \
            self.__upper_bound)
        self.__no_outliers_boolean = (self.__under_lower_bound_boolean | \
            self.__above_upper_bound_boolean).sum(axis=1)==0
        self.__data = self.__data[self.__no_outliers_boolean]
        self.old_data_shape = self.__data_shape
        self.__data_shape = self.__data.shape
        print('Dataset without outliers shape:', self.__data_shape)
        print('Reduced from ', self.old_data_shape,' to ', self.__data_shape, \
            '[-',np.round((self.old_data_shape[0] - self.__data_shape[0])/ \
                self.old_data_shape[0],decimals=3),'%]')
        if self.__class_name in self.__data.columns:
            return self.__data.drop(self.__class_name), \
                self.__target[self.__no_outliers_boolean]
        else:
            return self.__data, self.__target[self.__no_outliers_boolean]


    def standardize(self, flag_update_data=False):
        scaler = StandardScaler()
        data = scaler.fit_transform(self.__data)
        data = pd.DataFrame(data, columns = self.__header)
        if flag_update_data:
            self.__data = data

        return data

    def normalize(self, flag_update_data=False):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(self.__data)
        data = pd.DataFrame(data, columns = self.__header)
        if flag_update_data:
            self.__data = data

        return data

    def data_update(self, data):
        self.__data = data
        self.__header = self.__data.columns
        self.__data_shape = self.__data.shape

    
def grid_plot(data, class_name):
    grid_plot = sns.PairGrid(data, hue=class_name)
    grid_plot.map_diag(sns.histplot)
    grid_plot.map_offdiag(sns.scatterplot)
    grid_plot.add_legend()

def scatter_plot(data, target, figure_size):
    features_number = data.shape[1]
    sqrt_features_number = int(np.ceil(np.sqrt(features_number)))
    unique_classes = target.unique()
    figure, ax = plt.subplots(figsize=figure_size, nrows = \
        sqrt_features_number, ncols = sqrt_features_number)
    ax = ax.flatten()
    for feat in range(features_number):
        for label in unique_classes:
            x_axis = data[data.columns[feat]][target==label].values
            y_axis = target[target==label].values
            ax[feat].scatter(x_axis, y_axis)
            ax[feat].set_title(data.columns[feat])
            ax[feat].legend(unique_classes)
    plt.tight_layout()

def tic():
    '''Functions used to compute the time spent during a process
    (Homemade version of matlab tic and toc functions)'''
    global start_time_for_tictoc
    start_time_for_tictoc = time.time()

def toc():
    if 'start_time_for_tictoc' in globals():
        print ('\nElapsed time is ')
        print (str(time.time() - start_time_for_tictoc))
        print('seconds.\n')
    else:
        print ('\nToc: start time not set\n')

def return_toc():
    import time
    if 'start_time_for_tictoc' in globals():
        tt = time.time() - start_time_for_tictoc
    else:
        print ('\nToc: start time not set\n')
    return tt