import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

###################### General functions ######################

def scale_general(df, scaler):
    ''' Scale a dataframe using a given scaler (fit and transform).
        Keeps index and column names.
        Return new dataframe, scaler.
        
        Args:
        - df : pandas dataframe
        - scaler : initialized sklearn scaler function
        
        return scaled df and fit scaler
    '''
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler


# To use with test data if fitted on train data
def transform_df(df, scaler):
    ''' Scale a dataframe using a fit scaler.
        This is to prevent data leakage when the fit and transform datasets are different.
        Keeps index and column names.
        Return new dataframe.
        
        Args:
        - df : pandas dataframe
        - scaler : initialized and fit sklearn scaler function
        
        return scaled df
    '''
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    return df_scaled


def calc_error_bounds(df, measures_list, errors_list):
    ''' Get lower and upper error bounds of a measurement in a data frame
    Args: 
    df: pandas dataframe containing measures and errors, 
    measures_list: list of column names with measured values
    errors_list: list of column names with error values (stdev or error, etc)

    returns two dataframes with measures with errors subtracted (lower boundary, with suffix "_LB") 
        and measures with error added (upper boundary, with suffix "_UB")
    '''
    df_measures = df[measures_list]
    df_errors = df[errors_list]
    df_lower_boundary = df_measures - df_errors.values
    df_lower_boundary = df_lower_boundary.add_suffix('_LB')
    df_upper_boundary = df_measures + df_errors.values
    df_upper_boundary = df_upper_boundary.add_suffix('_UB')
    
    return df_lower_boundary, df_upper_boundary


def findAUC(df, A, B, p, f_init=8, f_end=22, name=''):
    '''Calculate the AUC for attenuation measurement and add it to the current dataframe
    Args:
    - df : pandas dataframe
    - A : A parameter as a column
    - B : B parameter as a column
    - p : p parameter as a column
    - f_init : lowest value on the curve
    - f_end : highest value on the curve
    - name : name of new column
    return updated dataframe
    '''

    def polyFunc(f):
        return A/5*f**5 + B/3*f**3 + p*f
    df_AUC = polyFunc(f_end) - polyFunc(f_init)
    df_AUC = df_AUC.rename(name)
    df_AUC = df_AUC.astype('float64')
    return df_AUC


def biplot(pca, data, pc_i, pc_j,title, color='b', plot_vectors=True):
    '''
    pca - fit PCA
    data - data frame of original data
    pc_i - int, first principal component to look at
    pc_j- int, second principal component to look at
    title - title for plot
    color - optional, if passed should be a dict that maps the index to colors
    '''
    xvector = pca.components_[pc_i] 
    yvector = pca.components_[pc_j]
    
    xs = pca.transform(data)[:,pc_i] 
    ys = pca.transform(data)[:,pc_j]
    
    plt.figure(figsize=(10,10))
    if plot_vectors:
        for i in range(len(xvector)):
        # arrows project features (ie columns) as vectors onto PC axes
            plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                color='r', width=0.0005, head_width=0.0025)
            plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
                list(data.columns.values)[i], color='r',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    # Each point is plotted if no color is given
    if color == 'b':
        for i in range(len(xs)):
        # circles project samples (ie rows  as points onto PC axes
            plt.plot(xs[i], ys[i], 'bo')
            plt.text(xs[i]*0.86, ys[i]*0.95, list(data.index)[i], color=color)
    # color passed in should be a dict that maps the index to colors
    else:
        plt.scatter(xs, ys, color=[ color[i] for i in data.index ])
        markers = [plt.Line2D([0,0],[0,0],color=c, marker='o', linestyle='') for c in color.values()]
        plt.legend(markers, color.keys(), loc='upper right', numpoints=1)
    plt.xlabel("PC"+str(pc_i+1))
    plt.ylabel("PC"+str(pc_j+1))
    plt.title(title)
    plt.show()

def load_data(path):
    '''
    load_data for consistency columns in analyses.
    - path : path to csv file

    '''
    df = pd.read_csv(path, index_col=-2)
    df['log_MS_Avg'] = np.log(df['MS_Avg'])
    df['log_beta_avg'] = np.log(df['Beta_avg'])
    
    df.drop(columns=['Beta_avg', 'MS_Avg'], inplace=True)
    df = df.loc[:, regression_cols]
    df = scale_general(df, MinMaxScaler())[0]
    return df


###################### Lists for handling dataframes ######################

error_cols = ["MS_neg_error", "MS_pos_error", "TEP_error"]
regression_cols = ['KJIC', 'log_beta_avg', 'TEP_average', 'log_MS_Avg', 'PC_IF_2.25MHz', 'PC_IF_3.5MHz','PC_BS']
regression_cols_real_data = ['KJIC', 'TEP_average', 'PC_IF_2.25MHz', 'PC_IF_3.5MHz','PC_BS']
