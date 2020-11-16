import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

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
    if plot_vectors:
        for i in range(len(xvector)):
        # arrows project features (ie columns) as vectors onto PC axes
            plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                color='r', width=0.0005, head_width=0.0025)
            plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
                list(data.columns.values)[i], color='r',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.xlabel("PC"+str(pc_i+1))
    plt.ylabel("PC"+str(pc_j+1))
    plt.title(title)
    plt.show()

def load_data(path, scaler):
    '''
    load_data for consistency columns in analyses.
    Args:
    - path : path to csv file
    - scaler :sklearn scaler

    '''

    df = pd.read_csv(path, index_col=-1)
    df['log_MS_Avg'] = np.log(df['MS_Avg'])
    df['log_beta_avg'] = np.log(df['Beta_avg'])    
    df.drop(columns=['Beta_avg', 'MS_Avg'], inplace=True)
    df = df.loc[:, regression_cols]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['KJIC']), 
                                                                df['KJIC'], 
                                                                test_size=0.2, 
                                                                random_state=2020)
    X_train, scaler = scale_general(X_train, scaler)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return  X_train, X_test, y_train, y_test


def plot_corr(data, figsize=(15,15)):
    '''
    Plot correlation 
    Args:
    - data: pd dataframe
    '''
    corr = data.corr()
    sns.set(font_scale=1.2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(corr, mask=mask, square=True, 
                         vmin= -1, vmax=1,
                         cmap='RdBu_r', center=0, annot=True,
                        annot_kws={'fontsize':8})
        
def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(y['Observed'], y['F_Time'], prediction)
    return result[0]

def plot_performance(gcv):
    n_splits = gcv.cv.n_splits
    cv_scores = {"alpha": [], "test_score": [], "split": []}
    order = []
    for i, params in enumerate(gcv.cv_results_["params"]):
        name = "%.5f" % params["alpha"]
        order.append(name)
        for j in range(n_splits):
            vs = gcv.cv_results_["split%d_test_score" % j][i]
            cv_scores["alpha"].append(name)
            cv_scores["test_score"].append(vs)
            cv_scores["split"].append(j)
    df = pandas.DataFrame.from_dict(cv_scores)
    _, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x="alpha", y="test_score", data=df, order=order, ax=ax)
    _, xtext = plt.xticks()
    for t in xtext:
        t.set_rotation("vertical")

###################### Lists for handling dataframes ######################

survival_cols = ['Observed','F_Time']
feature_selection = ['NLE_ratio_85_17','amp_ratio','pos_ratio','NDE_cycle','NLO_avg']
feature_selection2 = ['NLE_ratio_85_17','amp_ratio','pos_ratio','Avg_RP','NLO_avg']
