import pandas as pd
import matplotlib.pyplot as plt

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

def updated_df(df, measures_list, errors_list):
    data = []
    AUC = findAUC(df, A=df['A'], B=df['B'], p=df['p'], name='AUC_avg')
    error = calc_error_bounds(df, measures_list, errors_list)
    
    AUC = findAUC(df, A=df['A'], B=df['B'], p=df['p'], name='AUC_avg')
    AUC_LB = findAUC(error[0], A=error[0]['A_LB'], B=error[0]['B_LB'], p=error[0]['p_LB'], name='AUC_LB')
    AUC_UB = findAUC(error[1], A=error[1]['A_UB'], B=error[1]['B_UB'], p=error[1]['p_UB'], name='AUC_UB')
    
    AUC_STD = (AUC_UB - AUC_LB)*0.5
    AUC_STD = AUC_STD.rename('AUC_std')
    AUC_STD = AUC_STD.astype('float64')

    CF_perm = df['mean_CF']/df['mean_perm']
    CF_perm = CF_perm.rename('CF_perm')
    CF_perm = CF_perm.astype('float64')
    
    CF_perm_std = df['std_CF']/df['std_perm']
    CF_perm_std = CF_perm_std.rename('CF_perm_std')
    CF_perm_std = CF_perm_std.astype('float64')
    
    data = pd.concat([df, error[0], error[1], AUC, AUC_LB, AUC_UB, AUC_STD, CF_perm, CF_perm_std], axis=1)
    return data
    
def get_subsample_df(df):
    '''Create dataframe for all 4 types of sample (tubes, pipes, tubes identified, tubes blind)
    Args:
    -df : pmadas dataframe
    return df tube, pipe, tubes identifies, tubes, blind
    '''
    tube_df = df.copy()[df.index.str.contains('T_')]
    tube_df.dropna(how='all', axis=1, inplace=True)
    
    pipe_df = df.copy()[df.index.str.contains('P_')]
    pipe_df.dropna(how='all', axis=1, inplace=True)
    
    tube_wo_blind_df = df.copy()[~df.index.str.contains('T_B|P')]
    tube_wo_blind_df.dropna(how='all', axis=1, inplace=True)
    
    tube_blind_df = df.copy()[df.index.str.contains('T_B')]
    tube_blind_df.dropna(how='all', axis=1, inplace=True)
    
    return tube_df, pipe_df, tube_wo_blind_df, tube_blind_df
 

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


###################### Lists for handling dataframes ######################

drop_list_absorption_500_200 = ['Absorption_avg_500','Absorption_std_500','Absorption_avg_200','Absorption_std_200']

drop_list_absorption_100 = ['Absorption_avg_100','Absorption_std_100']

measures_list = ['TEP_mean_uV_C','Absorption_avg_500', 'Absorption_avg_50', 'Absorption_avg_100',
                 'backscatter_avg', 'A', 'B', 'p', 'Absorption_avg_200', 'mean_CF','mean_perm', 'mean_MBN','mean_CF_g', 'mean_perm_g','mean_pMBN_g']

errors_list = [ 'TEP_error_uV_C','Absorption_std_500',  'Absorption_std_50', 'Absorption_std_100', 'backscatter_std', 'A std', 'B std', 'p std', 'Absorption_std_200', 'std_CF','std_perm','std_MBN','std_CF_g','std_perm_g','std_pMBN_g']

without_std_g_list = ['TEP_mean_uV_C','Absorption_avg_500', 'Absorption_avg_50', 'Absorption_avg_100',
                 'backscatter_avg', 'A', 'B', 'p', 'Absorption_avg_200', 'mean_CF','mean_perm', 'mean_MBN','TEP_error_uV_C','Absorption_std_500',  'Absorption_std_50', 'Absorption_std_100', 'backscatter_std', 'A std', 'B std', 'p std', 'Absorption_std_200', 'std_CF','std_perm','std_MBN',]


correlation_list = ['TEP_mean_uV_C', 'backscatter_avg', 'Absorption_avg_50', 'Absorption_avg_100', 
       'CF_perm', 'AUC_avg']


correlation_std_list = ['TEP_error_uV_C', 'backscatter_std', 'Absorption_std_50', 'Absorption_std_100', 
       'CF_perm_std','AUC_std']

feat_mean2 = ['TEP_mean_uV_C', 'Absorption_avg_500','backscatter_avg', 'Absorption_avg_50', 'Absorption_avg_100', 'Absorption_avg_200', 'mean_CF', 'mean_perm', 'AUC_avg']

feat_stds2 = ['TEP_error_uV_C', 'Absorption_std_500','backscatter_std','Absorption_std_50', 
    'Absorption_std_100','Absorption_std_200', 'std_CF', 'std_perm', 'AUC_std']

feat_mean = ['TEP_mean_uV_C', 'Absorption_avg_500','backscatter_avg', 'Absorption_avg_50', 'Absorption_avg_100', 'Absorption_avg_200', 'CF_perm', 'AUC_avg']

feat_stds = ['TEP_error_uV_C', 'Absorption_std_500','backscatter_std','Absorption_std_50', 
    'Absorption_std_100','Absorption_std_200', 'CF_perm_std', 'AUC_std']



data_generation_values = ['TEP_mean_uV_C', 'Absorption_avg_500','backscatter_avg', 
       'Absorption_avg_50', 'A', 'B', 'p', 'Absorption_avg_100', 
       'Absorption_avg_200', 'mean_CF', 'mean_perm', 'mean_MBN']

data_generation_stds = ['TEP_error_uV_C', 'Absorption_std_500','backscatter_std','Absorption_std_50', 
    'A std','B std','p std', 'Absorption_std_100','Absorption_std_200', 'std_CF','std_perm', 'std_MBN',
    'std_CF_g']

minimal_informative_features = ["Absorption_avg_50","CF_perm","AUC_avg","backscatter_avg"]