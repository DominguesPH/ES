import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix  # matriz de confusão
from IPython.display import display
import time

class MyDataset:
    '''MyDataset class is used to load, analyze and preprocess a
    dataset. The original dataset is saved in "self.__original_data",
    while the manipulated one is saved in "self.__data".
    Insert "filename" and "dataset_path" to initialize the class.'''
    def __init__(self, dataset_path, filename):
        self.__dataset_path = dataset_path
        self.__filename = filename

    def load_csv(self, skiprows=0, delimiter=';', \
        flag_return_data=False, class_name=None):
        data = pd.read_csv(os.path.join(self.__dataset_path, self.__filename), \
            skiprows = skiprows, delimiter = delimiter)
        self.__data = data
        self.__original_data = data
        self.__header =  self.__data.columns
        self.__data_shape = self.__data.shape
        print(self.__filename+' dataset was loaded!')
        print(self.__data_shape)
        if flag_return_data:
            return self.__data
        if class_name==None:
            self.__class_name = self.__header[-1]
        else:
            self.__class_name = class_name

    def load_txt(self, skiprows=0, delimiter=',', \
        flag_return_data=False, class_name=None):
        data = np.loadtxt(os.path.join(self.__dataset_path, self.__filename), \
            skiprows = skiprows, delimiter = delimiter)
        self.__header = np.arange(data.shape[1])
        data = pd.DataFrame(data, columns=self.__header)
        self.__data = data
        self.__original_data = data
        self.__data_shape = self.__data.shape
        print(self.__filename+' dataset was loaded!')
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
    
    def get_original_data(self):
        return self.__original_data

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
        self.__upper_bound = self.__quartile_3rd + \
            (1.5 * self.__interquartile)
        self.__lower_bound = self.__quartile_1st - \
            (1.5 * self.__interquartile)
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
        print('Reduced from ', self.old_data_shape,' to ', \
            self.__data_shape, '[-',np.round((self.old_data_shape[0] - \
            self.__data_shape[0]) / self.old_data_shape[0],decimals=3), '%]')
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


class MyResults:
    '''MyResults class is responsible for analyse the resultant
    dictionary gathered in the classification task'''
    def __init__(self, results_path, filename):
        self.__results_path = results_path
        self.__filename = filename
    
    def load(self):
        path_and_file = os.path.join(self.__results_path, self.__filename)
        with open(path_and_file, 'rb') as fname:
            self.__results = pk.load(fname)

    def get_results(self):
        return self.__results
    
    def display_results(self):
        '''display_results() show all the results sorted by dataset'''
        self.__results_dataframe = pd.DataFrame(self.__results)
        self.__ID = np.array(self.__results['ID'])
        self.__datasets = np.array(self.__results['Dataset'])
        self.__models = np.array(self.__results['Model'])
        for dataset in np.unique(self.__datasets):
            selected_ID = self.__ID[self.__datasets == dataset]
            output_dict = {}
            for id in selected_ID:
                model = self.__models[self.__ID==id][0]
                ID_CVResults = pd.DataFrame(pd.DataFrame(\
                    self.__results_dataframe[self.__results_dataframe['ID'] \
                    == id])['CVResults'][pd.DataFrame( \
                    self.__results_dataframe[self.__results_dataframe['ID']  \
                    == id])['CVResults'].index[0]])
                params = ID_CVResults['params'].values
                parameters = {}
                for k in params[0].keys():
                    parameters[k] = tuple(parameters[k]\
                        for parameters in params)
                mean_cv_score = ID_CVResults['mean_test_score']
                std_cv_score = ID_CVResults['std_test_score']
                model_output = { \
                    'ID': np.repeat(id, len(mean_cv_score)), \
                    'Dataset': np.repeat(dataset, len(mean_cv_score)), \
                    'Model': np.repeat(model, len(mean_cv_score)) \
                }
                model_output.update(parameters)
                model_output.update({'Mean_CV_Score': mean_cv_score})
                model_output.update({'Std_CV_Score': std_cv_score})
                model_output_dataframe = pd.DataFrame(model_output)
                display(model_output_dataframe)

    def results_dataframe(self, flag_output=True):
        registers = self.__results
        registers_ID = pd.Series(registers['ID'])
        registers_models = np.array(registers['Model'])
        registers_datasets_unique = np.unique(registers['Dataset'])
        output_dataframe = pd.DataFrame([], columns = ['ID', 'dataset', \
            'model', 'parameters','mean_cv_score', 'std_cv_score'])
        for dataset in registers_datasets_unique:
            selected_ID = np.array(registers['ID'])[np.array( \
                registers['Dataset'])==dataset]
            for id in selected_ID:
                ID_index = registers_ID[registers_ID==id].index
                model = registers_models[ID_index][0]
                registers_current_cvresults = np.array( \
                    registers['CVResults'])[ID_index]
                cvresults = {}
                for k in registers_current_cvresults[0].keys():
                    cvresults[k] = tuple(cvresults[k] \
                        for cvresults in registers_current_cvresults)
                mean_cv_score = cvresults['mean_test_score'][0]
                std_cv_score = cvresults['std_test_score'][0]
                parameters = np.array(registers_current_cvresults[0]\
                    ['params'])
                cross_validation_number = len(mean_cv_score)
                output_dataframe = output_dataframe.append(pd.concat([
                    pd.DataFrame(np.repeat(id, cross_validation_number), \
                        columns=['ID']),
                    pd.DataFrame(np.repeat(dataset, cross_validation_number), \
                        columns=['dataset']),
                    pd.DataFrame(np.repeat(model, cross_validation_number), \
                        columns=['model']),
                    pd.DataFrame(mean_cv_score, columns=['mean_cv_score']),
                    pd.DataFrame(std_cv_score, columns=['std_cv_score']),
                    pd.DataFrame(parameters, columns=['parameters']),
                ], axis = 1,), ignore_index = True)
        self.__results_dataframe = output_dataframe
        if flag_output:
            return self.__results_dataframe

    def general_best_results(self):
        '''Display best results in general'''
        results_dataframe = self.__results_dataframe
        maximum_mean_cv_score = results_dataframe['mean_cv_score'].max()
        maximum_index = find_indexes(results_dataframe, 'mean_cv_score', \
            maximum_mean_cv_score)
        print('The best results considering cross-validation mean score for', \
            'all conditions searched')
        display(results_dataframe.iloc[maximum_index])

    def datasets_best_results(self):
        '''Best results for each dataset'''
        results_dataframe = self.__results_dataframe
        all_datasets = results_dataframe['dataset'].unique()
        for dataset in all_datasets:
            dataset_index = find_indexes(results_dataframe, 'dataset', \
                dataset)
            results_for_one_dataset = results_dataframe.iloc[dataset_index]
            maximum_mean_cv_score = results_for_one_dataset\
                ['mean_cv_score'].max()
            maximum_dataset_cv_index = find_indexes(results_for_one_dataset, \
                'mean_cv_score', maximum_mean_cv_score)
            print('The best results considering cross-validation mean', \
                'score for', dataset, 'dataset')
            display(results_dataframe.iloc[maximum_dataset_cv_index])
    
    def datasets_comparison(self):
        sns.set(style="whitegrid",font_scale = 2)
        results_dataframe = self.__results_dataframe
        models_unique = results_dataframe['model'].unique()
        pal = sns.color_palette(n_colors = len(models_unique))
        datasets_comparison = ['Normalized', 'Standardized'] 
        for dataset in datasets_comparison:
            fig, ax = plt.subplots(figsize=(20,10), ncols=2, nrows=2)
            ax = ax.flatten()
            axis = 0
            for model in models_unique:
                current_dataframe = get_similar_datasets( \
                    results_dataframe, \
                    dataset, \
                    model)
                boxplot = sns.boxplot(x = 'model', y = 'mean_cv_score', \
                    hue = 'dataset', data = current_dataframe, \
                    ax = ax[axis])
                axis += 1
                boxplot.legend_.remove()
            plt.legend(bbox_to_anchor=(1, 1))
            plt.tight_layout()


class  MyPredictions:
    '''MyPredictions class is responsible for find and plot the predictions
    accordingly to the test 'iD' '''
    def __init__(self, results_path, filename):
        self.__results_path = results_path
        self.__filename = filename
    
    def load(self):
        path_and_file = os.path.join(self.__results_path, self.__filename)
        with open(path_and_file, 'rb') as fname:
            self.__results = pk.load(fname)

    def get_predictions(self):
        return self.__results

    def confusion_matrix_for_id(self, id):
        registers = self.__results
        registers_ID = pd.Series(registers['ID'])
        index = registers_ID.index[registers_ID==id][0]
        real_class = registers['Real_Label'][index].values
        predicted_class = registers['Predicted_Label'][index]
        print('Dataset:', registers['Dataset'][index])
        print('Model:', registers['Model'][index])
        print('Model\'s parameters:')
        display(pd.DataFrame(registers['Best_Parameters'][index], \
            index=['Values']))
        matrix_of_confusion = confusion_matrix(real_class, predicted_class, \
            labels=[0,1])
        confusion_dataframe = pd.DataFrame(matrix_of_confusion, index = \
            [i for i in "01"], columns = [i for i in "01"])
        colormap = plt.cm.viridis
        plt.figure(figsize = (4,3.3))
        graph = sns.heatmap(confusion_dataframe, annot=True, cmap=colormap, \
            fmt='d', linewidths=0.5)
        graph.set_xlabel('Predicted Class')
        graph.set_ylabel('Real Class')

def find_indexes(dataframe, column, equality_condition):
    return dataframe.index[dataframe[column] == equality_condition]

def get_similar_datasets(dataframe, word, model):
    all_similar = []
    for k in range(dataframe.shape[0]):
        all_similar.append(word in dataframe['dataset'].iloc[k])
    all_similar = dataframe.index[all_similar]
    similar_datasets = dataframe.iloc[all_similar]
    return similar_datasets[similar_datasets['model']==model]

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

def append_value(dict_obj, key, value):
    '''Function from https://bit.ly/3qj3kts '''
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value