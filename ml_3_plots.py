# -*- coding: utf-8 -*-
"""
Plot results of machine learning based data analysis
v120
@author: Dr. David Steyrl david.steyrl@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import shap
from itertools import combinations
from scipy.interpolate import interp1d
from scipy.stats import t
from sklearn.metrics import confusion_matrix


def lfp(path_load):
    """
    Returns pickle file at load path.

    Parameters
    ----------
    path_load : string
        Path to pickle file.

    Returns
    -------
    data : pickle
        Returns stored data.

    """
    # Load from pickle file
    with open(path_load, 'rb') as filehandle:
        # Load data from binary data stream
        data = pickle.load(filehandle)
    return data


def create_dir(path):
    """
    Create specified directiry if not existing.

    Parameters
    ----------
    path : string
        Path to to check to be created.

    Returns
    -------
    None.

    """
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)


def dep_two_sample_ttest(x1, x2, test_train_ratio, alpha, side='two'):
    """
    Implementation of the Nadeau and Bengio correction of dependent sample
    (due to cross-validation's resampling) two sample Student's t-test for
    unequal sample sizes and unequal variances aka Welch's test
    t_stat = mean(x1)-mean(x2)/sqrt(var_n_ratio1+var_n_ratio2)
    whereas var_n_ratiox (= var_samplesx/n_samplesx) is replaced by
    corrected var_n_ratiox (= (1/n_samplesx+test_train_ratio)*var_samples(x)
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366

    Parameters
    ----------
    x1 : ndarray
        Data of condition 1.
    x2 : ndarray
        Data of condition 2.
    test_train_ratio : float
        n_test_samples / n_training_samples.
    alpha : float
        Confidence level.
    side : sring ('one', 'two')
        If test is one or two sided. Default is two.

    Returns
    -------
    t_stat : float
        Corrected t statistic.
    df : float
        Degrees of freedom.
    cv : float
        Critical value for significance.
    p : float
        Corrected p value.

    """
    # Get n samples data 1
    n1 = len(x1)
    # Get n samples data 2
    n2 = len(x2)
    # Get corrected variance samples ratio of data 1
    var1_n1 = (1/n1+test_train_ratio)*np.var(x1)
    # Get corrected variance samples ratio of data 2
    var2_n2 = (1/n2+test_train_ratio)*np.var(x2)
    # Compute corrected t statistics
    t_stat = (np.mean(x1)-np.mean(x2))/np.sqrt(var1_n1+var2_n2)
    # degrees of freedom
    df = (var1_n1+var2_n2)**2/(((var1_n1)**2/(n1-1))+((var2_n2)**2/(n2-1)))
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    if 'one' in side:
        p = (1.0 - t.cdf(t_stat, df))
    elif 'two' in side:
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    else:
        print('Error: unknown side')
    # return everything
    return t_stat, df, cv, p


def print_fit_regression_scatter(task, results, plots_path):
    """
    Print model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # True values
    true_values = results['true_values']
    # True values flat
    true_values_flat = np.concatenate(true_values).flatten()
    # Predicted values
    pred_values = results['pred_values']
    # Predicted values flat
    pred_values_flat = np.concatenate(pred_values).flatten()
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Print data
    ax.scatter(true_values_flat,
               pred_values_flat,
               zorder=2,
               alpha=0.05)
    # Add optimal fit line
    ax.plot([-5000, 5000], [-5000, 5000], 'r', zorder=3)
    # Get true values range
    true_values_range = max(true_values_flat) - min(true_values_flat)
    # Set x-axis limits
    ax.set_xlim(min(true_values_flat) - true_values_range/20,
                max(true_values_flat) + true_values_range/20)
    # Get predicted values range
    pred_values_range = max(pred_values_flat) - min(pred_values_flat)
    # Set y-axis limits
    ax.set_ylim(min(pred_values_flat) - pred_values_range/20,
                max(pred_values_flat) + pred_values_range/20)
    # Set title
    ax.set_title(task['predictor_name']+' prediction of target: ' +
                 task['y_name'][0], fontsize=10)
    # Set xlabel
    ax.set_xlabel('Given true values', fontsize=10)
    # Set ylabel
    ax.set_ylabel('Predicted values', fontsize=10)

    # Add MAE -----------------------------------------------------------------
    # Extract MAE
    mae = [i[0][0] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i[0][0] for i in results['scores_sh']]
    # Calculate p-value between MAE and shuffle MAE
    _, _, _, pval_mae = dep_two_sample_ttest(
        np.array(mae_sh), np.array(mae),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Add MAE results to plot
    ax.text(.35, .125, ('MAE original mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}').format(
            np.mean(mae),
            np.std(mae),
            np.median(mae)),
            transform=ax.transAxes,
            fontsize=10)
    # Add MAE p val results to plot
    ax.text(.35, .09, ('MAE shuffle mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}|p:{:.3f}').format(
            np.mean(mae_sh),
            np.std(mae_sh),
            np.median(mae_sh),
            pval_mae),
            transform=ax.transAxes,
            fontsize=10)

    # Add MSE -----------------------------------------------------------------
    # Extract MSE
    mse = [i[1][0] for i in results['scores']]
    # Extract MSE shuffle
    mse_sh = [i[1][0] for i in results['scores_sh']]
    # Calculate p-value between MSE and shuffle MSE
    _, _, _, pval_mse = dep_two_sample_ttest(
        np.array(mse_sh), np.array(mse),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Add MSE results to plot
    ax.text(.35, .055, ('MSE original mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}').format(
            np.mean(mse),
            np.std(mse),
            np.median(mse)),
            transform=ax.transAxes,
            fontsize=10)
    # Add MSE p val results to plot
    ax.text(.35, .02, ('MSE shuffle mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}|p:{:.3f}').format(
            np.mean(mse_sh),
            np.std(mse_sh),
            np.median(mse_sh),
            pval_mse),
            transform=ax.transAxes,
            fontsize=10)

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i[2][0] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i[2][0] for i in results['scores_sh']]
    # Calculate p-value between R² and shuffle R²
    _, _, _, pval_r2 = dep_two_sample_ttest(
        np.array(r2), np.array(r2_sh),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Add R² results to plot
    ax.text(.02, .96, ('R² original mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}').format(
            np.mean(r2),
            np.std(r2),
            np.median(r2)),
            transform=ax.transAxes,
            fontsize=10)
    # Add R² p val results to plot
    ax.text(.02, .925, ('R² shuffle mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}|p:{:.3f}').format(
            np.mean(r2_sh),
            np.std(r2_sh),
            np.median(r2_sh),
            pval_r2),
            transform=ax.transAxes,
            fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_fit_scatter_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


def print_fit_regression_violin(task, results, plots_path):
    """
    Print model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Extract MAE
    mae = [i[0][0] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i[0][0] for i in results['scores_sh']]
    # Extract MSE
    mse = [i[1][0] for i in results['scores']]
    # Extract MSE shuffle
    mse_sh = [i[1][0] for i in results['scores_sh']]
    # Extract R²
    r2 = [i[2][0] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i[2][0] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'MAE': pd.Series(np.array(mae)),
         'MSE': pd.Series(np.array(mse)),
         'R2': pd.Series(np.array(r2)),
         'Data': pd.Series(['original' for _ in mae]),
         'Dummy': pd.Series(np.ones(np.array(mae).shape).flatten())})
    # Save scores
    scores_df.to_csv(plots_path+'/'+task['predictor_name']+'_scores_' +
                     task['y_name'][0]+'.csv')
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'MAE': pd.Series(np.array(mae_sh)),
         'MSE': pd.Series(np.array(mse_sh)),
         'R2': pd.Series(np.array(r2_sh)),
         'Data': pd.Series(['shuffle' for _ in mae_sh]),
         'Dummy': pd.Series(np.ones(np.array(mae_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['MAE', 'MSE', 'R2']
    # Calculate p-value between MAE and shuffle MAE
    _, _, _, pval_mae = dep_two_sample_ttest(
        np.array(mae_sh), np.array(mae),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Calculate p-value between MSE and shuffle MSE
    _, _, _, pval_mse = dep_two_sample_ttest(
        np.array(mse_sh), np.array(mse),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Calculate p-value between R² and shuffle R²
    _, _, _, pval_r2 = dep_two_sample_ttest(
        np.array(r2), np.array(r2_sh),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Combine p-values into list
    p_vals = [pval_mae, pval_mse, pval_r2]
    # Make figure
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
    # Set tight figure layout
    fig.tight_layout()
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw='scott', cut=2, scale='width', gridsize=100,
                       width=0.8, inner='point', orient='h', linewidth=1,
                       saturation=0.75, ax=ax[i])
        # Add title if top plot
        if i == 0:
            # Add title
            ax[i].set_title(task['predictor_name']+' predition of target: ' +
                            task['y_name'][0], fontsize=10)
        # Remove legend
        if i > 0:
            ax[i].legend().remove()
        # Remove x label
        ax[i].set_xlabel('')
        # Set y label
        ax[i].set_ylabel(metric)
        # Remove y ticks
        ax[i].get_yaxis().set_ticks([])
        # Add significance signs
        if p_vals[i]*len(p_vals) < 0.05:
            ax[i].text(0.02, 0.4,
                       '**',
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=20,
                       transform=ax[i].transAxes)
        elif p_vals[i] < 0.05:
            ax[i].text(0.02, 0.4,
                       '*',
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=20,
                       transform=ax[i].transAxes)
    # Add significance text
    ax[-1].text(.60, .03, '*  p-value<0.05 \n** p-value<0.05 incl. Bonferroni',
                fontsize=10, transform=ax[-1].transAxes)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_fit_reg_violin_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()


def print_fit_classification_confusion(task, results, plots_path):
    """
    Print model fit as confusion matrix (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # True values
    true_values = results['true_values']
    # Predicted values
    pred_values = results['pred_values']
    # Sample weights  list
    sample_weights = [i[1] for i in results['scores']]
    # Accuracy
    acc = [i[0][0] for i in results['scores']]
    # Schuffle accuracy
    acc_sh = [i[0][0] for i in results['scores_sh']]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values).flatten()).tolist()

    # Make confusion matrix ---------------------------------------------------
    # Loop over single model results
    for c_true, c_pred, c_w in zip(true_values, pred_values, sample_weights):
        if 'con_mat' not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                c_true,
                c_pred,
                labels=class_labels,
                sample_weight=np.array([c_w[i] for i in c_true]),
                normalize='all')
        else:
            # Add confusion matrix
            con_mat = np.add(con_mat, confusion_matrix(
                c_true,
                c_pred,
                labels=class_labels,
                sample_weight=np.array([c_w[i] for i in c_true]),
                normalize='all'))
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(figsize=(3, 3))
    # Plot confusion matrix
    sns.heatmap(con_mat_norm*100,
                vmin=None,
                vmax=None,
                cmap=None,
                center=None,
                robust=True,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 10},
                linewidths=.5,
                linecolor='white',
                cbar=False,
                cbar_kws=None,
                square=True,
                xticklabels='auto',
                yticklabels='auto',
                mask=None,
                ax=ax)
    # Add y label to plot
    plt.ylabel('True Label', fontsize=10)
    # Add x label to plot
    plt.xlabel('Predicted Label', fontsize=10)
    # Calculate p-value between accuracy and shuffle accuracy
    _, _, _, pval_acc = dep_two_sample_ttest(
        np.array(acc), np.array(acc_sh),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05)
    # Add title
    plt.title(
        (str(task['predictor_name'])+' prediction of '+task['y_name'][0] +
         '\n'+'Accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
         '{:.2f}|{:.2f}'+'\n' +
         'Accuracy shuffle mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
         '{:.2f}|{:.2f}'+'\n' +
         'p-value: {:.3f}'+'\n' +
         '-- Confusion Matrix --').format(
            np.mean(acc)*100,
            np.std(acc)*100,
            np.median(acc)*100,
            np.mean(acc_sh)*100,
            np.std(acc_sh)*100,
            np.median(acc_sh)*100,
            pval_acc),
        fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_scores_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()


def print_fit_classification_violin(task, results, plots_path):
    """
    Print model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Extract accuracy
    acc = [i[0][0] for i in results['scores']]
    # Extract shuffle accuracy
    acc_sh = [i[0][0] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'ACC': pd.Series(np.array(acc)),
         'Data': pd.Series(['original' for _ in acc]),
         'Dummy': pd.Series(np.ones(np.array(acc).shape).flatten())})
    # Save scores
    scores_df.to_csv(plots_path+'/'+task['predictor_name']+'_scores_' +
                     task['y_name'][0]+'.csv')
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'ACC': pd.Series(np.array(acc_sh)),
         'Data': pd.Series(['shuffle' for _ in acc_sh]),
         'Dummy': pd.Series(np.ones(np.array(acc_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['ACC']
    # Calculate p-value between ACC and shuffle ACC
    _, _, _, pval_acc = dep_two_sample_ttest(
        np.array(acc), np.array(acc_sh),
        task['test_size_frac']/(1-task['test_size_frac']), 0.05, side='one')
    # Combine p-values into list
    p_vals = [pval_acc]
    # Make figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2.5))
    # Put ax into list
    ax = [ax]
    # Set tight figure layout
    fig.tight_layout()
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw='scott', cut=2, scale='width', gridsize=100,
                       width=0.8, inner='point', orient='h', linewidth=1,
                       saturation=0.75, ax=ax[i])
        # Add title if top plot
        if i == 0:
            # Add title
            ax[i].set_title(task['predictor_name']+' prediction of target: ' +
                            task['y_name'][0], fontsize=10)
        # Remove legend
        if i > 0:
            ax[i].legend().remove()
        # Remove x label
        ax[i].set_xlabel('')
        # Set y label
        ax[i].set_ylabel(metric)
        # Remove y ticks
        ax[i].get_yaxis().set_ticks([])
        # Add significance signs
        if p_vals[i]*len(p_vals) < 0.05:
            ax[i].text(0.02, 0.4,
                       '**',
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=20,
                       transform=ax[i].transAxes)
        elif p_vals[i] < 0.05:
            ax[i].text(0.02, 0.4,
                       '*',
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=20,
                       transform=ax[i].transAxes)
    # Add significance text
    ax[-1].text(.60, .03, '*  p-value<0.05 \n** p-value<0.05 incl. Bonferroni',
                fontsize=10, transform=ax[-1].transAxes)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_fit_clf_violin_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    plt.show()


def print_model_importance(task, m_imp, m_imp_sh, plots_path, class_label=-99):
    """
    Plot single model based feature importance.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    m_imp : list of ndarray
        List holding the model based feature importances per repetition of the
        cross-validation loop.
    m_imp_sh : list of ndarray
        List holding the model based feature importances per repetition of the
        cross-validation loop as obtained from shuffled data.
    plots_path : string
        Path to the plots.
    class_label : integer, optional
        Seperate model based feature importances by given class label.
        The default is -99.

    Returns
    -------
    None.

    """
    # Make model importance dataframe
    m_imp_df = pd.DataFrame(m_imp, columns=task['x_names'])
    # Make shuffle data model importance dataframe
    m_imp_sh_df = pd.DataFrame(m_imp_sh, columns=task['x_names'])
    # Sorting index
    if 'EN' in task['predictor_name'] or 'LR' in task['predictor_name']:
        # Sorting index by median absolute value of columns
        i_srt = m_imp_df.median().abs().sort_values(ascending=False).index
    elif ('ET' in task['predictor_name'] or 'RF' in task['predictor_name'] or
          'GB' in task['predictor_name']):
        # Sorting index by median value of columns
        i_srt = m_imp_df.median().sort_values(ascending=False).index
    else:
        print('Error: unknown predictor')
    # Sort model importance dataframe
    m_imp_sort_df = m_imp_df.reindex(i_srt, axis=1)
    # Sort shuffle model importance dataframe
    m_imp_sh_sort_df = m_imp_sh_df.reindex(i_srt, axis=1)
    # Add data origin to model importance dataframe
    m_imp_sort_df['Data'] = pd.DataFrame(
        ['original' for _ in range(m_imp_sort_df.shape[0])], columns=['Data'])
    # Add data origin to shuffle model importance dataframe
    m_imp_sh_sort_df['Data'] = pd.DataFrame(
        ['shuffle' for _ in range(m_imp_sh_sort_df.shape[0])],
        columns=['Data'])
    # Get value name
    if 'EN' in task['predictor_name'] or 'LR' in task['predictor_name']:
        value_name = 'Model weights'
    elif ('ET' in task['predictor_name'] or 'RF' in task['predictor_name']):
        if 'reg' in task['kind']:
            value_name = 'Reduction in MSE'
        elif 'clf' in task['kind']:
            value_name = 'Reduction in Gini index'
    elif 'GB' in task['predictor_name']:
        value_name = 'Reduction in MSE'
    else:
        print('Error: unknown predictor')
    # Melt model importance dataframe
    m_imp_sort_melt_df = m_imp_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Melt shuffle model importance dataframe
    m_imp_sh_sort_melt_df = m_imp_sh_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Concatenate importances dataframes
    all_m_imp_sort_melt_df = pd.concat([m_imp_sort_melt_df,
                                        m_imp_sh_sort_melt_df], axis=0)
    # Make figure
    fig, ax = plt.subplots(figsize=(5, 1+m_imp_df.shape[1]*0.6))
    # Plot data
    ax = sns.violinplot(x=value_name, y='predictors', hue='Data',
                        data=all_m_imp_sort_melt_df, bw='scott', cut=2,
                        scale='width', gridsize=100, width=0.8, inner='point',
                        orient='h', linewidth=1, saturation=0.75, ax=ax)
    # Set legend position
    plt.legend(loc='lower right')
    # Make title string
    if class_label == -99:
        title_str = (task['predictor_name'] +
                     ' model feature importance for target: ' +
                     task['y_name'][0])
    elif class_label > -99:
        title_str = (task['predictor_name'] +
                     ' model feature importance for target: ' +
                     task['y_name'][0]+' | class: '+str(class_label))
    # Add title
    ax.set_title(title_str, fontsize=10)

    # Add significance to the plot --------------------------------------------
    # Initialize pval feature importance list
    pval_m_imp = []
    # Loop over importances
    for i, _ in enumerate(task['x_names']):
        if ('EN' in task['predictor_name'] or 'LR' in task['predictor_name']):
            # Calculate the p-value
            _, _, _, c_pval = dep_two_sample_ttest(
                m_imp_sort_df.iloc[:, i], m_imp_sh_sort_df.iloc[:, i],
                task['test_size_frac']/(1-task['test_size_frac']), 0.05,
                side='one')
        elif ('ET' in task['predictor_name'] or
              'RF' in task['predictor_name'] or
              'GB' in task['predictor_name']):
            # Calculate the p-value
            _, _, _, c_pval = dep_two_sample_ttest(
                m_imp_sort_df.iloc[:, i], m_imp_sh_sort_df.iloc[:, i],
                task['test_size_frac']/(1-task['test_size_frac']), 0.05,
                side='one')
        else:
            print('Error: unknown predictor')
        # Append results
        pval_m_imp.append(c_pval)
    # Add significance text
    ax.text(.35, .02, '*  p-value<0.05 \n** p-value<0.05 incl. Bonferroni',
            fontsize=10, transform=ax.transAxes)
    # Add significance signs
    for i, c_pval in enumerate(pval_m_imp):
        if c_pval*len(pval_m_imp) < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_m_imp)),
                    '**',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)
        elif c_pval < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_m_imp)),
                    '*',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)

    # Save plots and results --------------------------------------------------
    # Make save path
    if class_label == -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_m_imp_' +
                     task['y_name'][0])
    elif class_label > -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_m_imp_' +
                     task['y_name'][0]+'_class_'+str(class_label))
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()
    # Save model importance dataframe
    m_imp_sort_df.to_csv(save_path+'.csv')
    # Make p-values dataframe
    pval_m_imp_df = pd.DataFrame(np.reshape(pval_m_imp, (1, -1)),
                                 columns=list(m_imp_sort_df.columns)[:-1])
    # Save p values
    pval_m_imp_df.to_csv(save_path+'_p_val.csv')


def print_all_model_importances(task, results, plots_path):
    """
    Plot all model based feature importances.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Model based importance
    m_imp = np.array(results['m_importances'])
    # Shuffle model based importance
    m_imp_sh = np.array(results['m_importances_sh'])
    # print model importance
    if len(m_imp.shape) > 2:
        # Get unique classes
        unique_classes = np.unique(np.concatenate(results['true_values']))
        # Iterate over unique classes
        for i, c_class in enumerate(unique_classes):
            print_model_importance(task,
                                   m_imp[:, i, :],
                                   m_imp_sh[:, i, :],
                                   plots_path,
                                   class_label=c_class)
    else:
        print_model_importance(task, m_imp, m_imp_sh, plots_path)


def print_permutation_importance(task, results, plots_path):
    """
    Print all permutation based feature importances.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Permutation based importance
    p_imp = np.array(results['p_importances'])
    # Shuffle data permutation based importance
    p_imp_sh = np.array(results['p_importances_sh'])
    # Permutation based importance dataframe
    p_imp_df = pd.DataFrame(p_imp, columns=task['x_names'])
    # Shuffle data permutation based importance dataframe
    p_imp_sh_df = pd.DataFrame(p_imp_sh, columns=task['x_names'])
    # Sorting index by median value of columns
    i_srt = p_imp_df.median().sort_values(ascending=False).index
    # Sort permutation importance dataframe
    p_imp_sort_df = p_imp_df.reindex(i_srt, axis=1)
    # Sort shuffle permutation importance dataframe
    p_imp_sh_sort_df = p_imp_sh_df.reindex(i_srt, axis=1)
    # Add data origin to model importance dataframe
    p_imp_sort_df['Data'] = pd.DataFrame(
        ['original' for _ in range(p_imp_sort_df.shape[0])], columns=['Data'])
    # Add data origin to shuffle model importance dataframe
    p_imp_sh_sort_df['Data'] = pd.DataFrame(
        ['shuffle' for _ in range(p_imp_sh_sort_df.shape[0])],
        columns=['Data'])
    # Get value name
    if 'reg' in task['kind']:
        value_name = 'Reduction of R²'
    elif 'clf' in task['kind']:
        value_name = 'Reduction of classification accuracy'
    else:
        print('Error: unknown task kind')
    # Melt permutation importance dataframe
    p_imp_sort_melt_df = p_imp_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Melt shuffle permutation importance dataframe
    p_imp_sh_sort_melt_df = p_imp_sh_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Concatenate importances dataframes
    all_p_imp_sort_melt_df = pd.concat([p_imp_sort_melt_df,
                                        p_imp_sh_sort_melt_df], axis=0)
    # Make figure
    fig, ax = plt.subplots(figsize=(5, 1+p_imp_df.shape[1]*0.6))
    # Plot data
    ax = sns.violinplot(x=value_name, y='predictors', hue='Data',
                        data=all_p_imp_sort_melt_df, bw='scott', cut=2,
                        scale='width', gridsize=100, width=0.8, inner='point',
                        orient='h', linewidth=1, saturation=0.75, ax=ax)
    # Set legend position
    plt.legend(loc='lower right')
    # Add title
    ax.set_title(task['predictor_name'] +
                 ' permutation feature importance for target: ' +
                 task['y_name'][0], fontsize=10)

    # Add significance to the plot --------------------------------------------
    # Initialize pval feature importance list
    pval_p_imp = []
    # Loop over importances
    for i, _ in enumerate(task['x_names']):
        # Calculate the p-value
        _, _, _, c_pval = dep_two_sample_ttest(
            p_imp_sort_df.iloc[:, i],
            p_imp_sh_sort_df.iloc[:, i],
            task['test_size_frac']/(1-task['test_size_frac']), 0.05,
            side='one')
        # Append results
        pval_p_imp.append(c_pval)
    # Add significance text
    ax.text(.35, .02, '*  p-value<0.05 \n** p-value<0.05 incl. Bonferroni',
            fontsize=10, transform=ax.transAxes)
    # Add significance signs
    for i, c_pval in enumerate(pval_p_imp):
        if c_pval*len(pval_p_imp) < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_p_imp)),
                    '**',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)
        elif c_pval < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_p_imp)),
                    '*',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)

    # Save plots and results --------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_p_imp_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()
    # Save permutation importance dataframe
    p_imp_sort_df.to_csv(save_path+'.csv')
    # Make p-values dataframe
    pval_p_imp_df = pd.DataFrame(np.reshape(pval_p_imp, (1, -1)),
                                 columns=list(p_imp_sort_df.columns)[:-1])
    # Save p values
    pval_p_imp_df.to_csv(save_path+'_p_val.csv')


def print_shap_importance(task, s_imp, s_imp_sh, plots_path, class_label=-99):
    """
    Print single SHAP based feature importance.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    s_imp : list of ndarray
        List holding the SHAP based feature importances per repetition of the
        cross-validation loop.
    s_imp_sh : list of ndarray
        List holding the SHAP based feature importances per repetition of the
        cross-validation loop as obtained from shuffled data.
    plots_path : string
        Path to the plots.
    class_label :  integer, optional
        Seperate model based feature importances by given class label.
        The default is -99.

    Returns
    -------
    None.

    """
    # SHAP importance
    s_imp_mean = np.array([np.mean(np.abs(c_imp), axis=0) for c_imp in s_imp])
    # Shuffle data SHAP importance
    s_imp_sh_mean = np.array([np.mean(np.abs(c_imp), axis=0) for c_imp in
                              s_imp_sh])
    # SHAP importance dataframe
    s_imp_df = pd.DataFrame(s_imp_mean, columns=task['x_names'])
    # Shuffle data SHAP importance dataframe
    s_imp_sh_df = pd.DataFrame(s_imp_sh_mean, columns=task['x_names'])
    # Sorting index by median value of columns
    i_srt = s_imp_df.median().sort_values(ascending=False).index
    # Sort SHAP importance dataframe
    s_imp_sort_df = s_imp_df.reindex(i_srt, axis=1)
    # Sort shuffle SHAP importance dataframe
    s_imp_sh_sort_df = s_imp_sh_df.reindex(i_srt, axis=1)
    # Add data origin to SHAP importance dataframe
    s_imp_sort_df['Data'] = pd.DataFrame(
        ['original' for _ in range(s_imp_sort_df.shape[0])], columns=['Data'])
    # Add data origin to shuffle SHAP importance dataframe
    s_imp_sh_sort_df['Data'] = pd.DataFrame(
        ['shuffle' for _ in range(s_imp_sh_sort_df.shape[0])],
        columns=['Data'])
    # Get value name
    value_name = 'mean(|SHAP|)'
    # Melt SHAP importance dataframe
    s_imp_sort_melt_df = s_imp_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Melt shuffle SHAP importance dataframe
    s_imp_sh_sort_melt_df = s_imp_sh_sort_df.melt(
        id_vars=['Data'], var_name=['predictors'],
        value_name=value_name)
    # Concatenate importances dataframes
    all_s_imp_sort_melt_df = pd.concat([s_imp_sort_melt_df,
                                        s_imp_sh_sort_melt_df], axis=0)
    # Make figure
    fig, ax = plt.subplots(figsize=(5, 1+s_imp_df.shape[1]*0.6))
    # Plot data
    ax = sns.violinplot(x=value_name, y='predictors', hue='Data',
                        data=all_s_imp_sort_melt_df, bw='scott', cut=2,
                        scale='width', gridsize=100, width=0.8, inner='point',
                        orient='h', linewidth=1, saturation=0.75, ax=ax)
    # Set legend position
    plt.legend(loc='lower right')
    # Make title string
    if class_label == -99:
        title_str = (task['predictor_name'] +
                     ' SHAP feature importance for target: ' +
                     task['y_name'][0])
    elif class_label > -99:
        title_str = (task['predictor_name'] +
                     ' SHAP feature importance for target: ' +
                     task['y_name'][0]+' | class: '+str(class_label))
    # Add title
    ax.set_title(title_str, fontsize=10)

    # Add significance to the plot --------------------------------------------
    # Initialize pval SHAP importance list
    pval_s_imp = []
    # Loop over importances
    for i, _ in enumerate(task['x_names']):
        # Calculate the p-value
        _, _, _, c_pval = dep_two_sample_ttest(
            s_imp_sort_df.iloc[:, i],
            s_imp_sh_sort_df.iloc[:, i],
            task['test_size_frac']/(1-task['test_size_frac']), 0.05,
            side='one')
        # Append results
        pval_s_imp.append(c_pval)
    # Add significance text
    ax.text(.35, .02, '*  p-value<0.05 \n** p-value<0.05 incl. Bonferroni',
            fontsize=10, transform=ax.transAxes)
    # Add significance signs
    for i, c_pval in enumerate(pval_s_imp):
        if c_pval*len(pval_s_imp) < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_s_imp)),
                    '**',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)
        elif c_pval < 0.05:
            ax.text(0.02, 1-((i+0.7)/len(pval_s_imp)),
                    '*',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    transform=ax.transAxes)

    # Save plots and results --------------------------------------------------
    # Make save path
    if class_label == -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_s_imp_' +
                     task['y_name'][0])
    elif class_label > -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_s_imp_' +
                     task['y_name'][0]+'_class_'+str(class_label))
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    plt.show()
    # Save model importance dataframe
    s_imp_sort_df.to_csv(save_path+'.csv')
    # Make p-values dataframe
    pval_s_imp_df = pd.DataFrame(np.reshape(pval_s_imp, (1, -1)),
                                 columns=list(s_imp_sort_df.columns)[:-1])
    # Save p values
    pval_s_imp_df.to_csv(save_path+'_p_val.csv')


def print_all_shap_importances(task, results, plots_path):
    """
    Plot all SHAP based feature importances.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate(results['true_values']))
    # SHAP based importance
    s_imp = [c_rep[0] for c_rep in results['s_importances']]
    # Shuffle SHAP based importance
    s_imp_sh = [c_rep[0] for c_rep in results['s_importances_sh']]
    # print SHAP importance
    if isinstance(s_imp[1], list):
        # Iterate over unique classes
        for i, c_class in enumerate(unique_classes):
            print_shap_importance(task,
                                  [c_rep[i] for c_rep in s_imp],
                                  [c_rep[i] for c_rep in s_imp_sh],
                                  plots_path,
                                  class_label=c_class)
    else:
        print_shap_importance(task, s_imp, s_imp_sh, plots_path)


def print_shap_values(task, shap_values, x_tst, plots_path, class_label=-99):
    """
    Print single SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    shap_values : ndarray
        Ndarray holding the SHAP values per tested sample.
    x_tst : ndarray
        X test data.
    plots_path : string
        Path to the plots.
    class_label : integer, optional
        Seperate model based feature importances by given class label.
        The default is -99.

    Returns
    -------
    None.

    """
    # Test set data DataFrame
    x_tst_df = pd.DataFrame(x_tst, columns=task['x_names'])
    # Plot shap values summary
    shap.summary_plot(shap_values,
                      features=x_tst_df,
                      feature_names=list(x_tst_df.columns),
                      max_display=len(task['x_names']),
                      alpha=.6,
                      color_bar=True,
                      show=False)
    # Colorbar aspect ratio
    plt.gcf().axes[-1].set_aspect(100)
    # Colorbar box aspect ratio
    plt.gcf().axes[-1].set_box_aspect(100)
    # Make title string
    if class_label == -99:
        title_str = (task['predictor_name'] +
                     ' SHAP values for target: ' +
                     task['y_name'][0])
    elif class_label > -99:
        title_str = (task['predictor_name'] +
                     ' SHAP values for target: ' +
                     task['y_name'][0]+' | class: '+str(class_label))
    # Add title
    plt.title(title_str)

    # Save plots and results --------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['predictor_name']+'_shape_values_' +
                 task['y_name'][0])
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    plt.show()


def print_all_shap_values(task, results, plots_path):
    """
    Plot all SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate(results['true_values']))
    # SHAP based importance
    s_imp = [c_rep[0] for c_rep in results['s_importances']]
    # Test set data
    x_tst = np.concatenate(
        [c_rep[1] for c_rep in results['s_importances']], axis=0)
    # print SHAP importance
    if isinstance(s_imp[1], list):
        # Iterate over unique classes
        for i, c_class in enumerate(unique_classes):
            print_shap_values(
                task,
                np.concatenate([c_rep[i] for c_rep in s_imp], axis=0),
                x_tst,
                plots_path, class_label=c_class)
    else:
        print_shap_values(
            task, np.concatenate(s_imp, axis=0), x_tst, plots_path)


def print_partial_dependence(task, grid, dep, y_range, feat, plots_path,
                             class_label=-99):
    """
    Plot single partial dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    grid : list
        List holding a grid per cross-validation repetition.
    dep : list
        List holding dependencies per cross-validation repetition.
    feat : string
        Name of the current feature.
    plots_path : string
        Path to the plots.
    class_label : integer, optional
        Seperate model based feature importances by given class label.
        The default is -99.

    Returns
    -------
    None.

    """
    # Make common grid
    grid_linspace = np.linspace(np.min(np.hstack(grid)),
                                np.max(np.hstack(grid)), 20)
    # Initiate
    dep_interp = []
    # Loop over repetitions of cv
    for i_rep, c_rep in enumerate(dep):
        # If more than 1
        if c_rep.shape[0] > 1:
            # Initiate interpolate
            inter = interp1d(grid[i_rep], dep[i_rep], kind='linear',
                             axis=- 1, copy=True, bounds_error=None,
                             fill_value='extrapolate', assume_sorted=True)
            # Interpolate rows to common grid
            dep_interp.append(inter(grid_linspace))
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Loop over cross validation repetitions
    for (c_grid, c_dep) in zip(grid, dep):
        # plot current grid vs. dependence
        ax.plot(c_grid, c_dep, alpha=0.1, color='C0', linestyle='solid',
                marker='*', markerfacecolor='C0', markersize=8)
    # Compute average dependency
    dep_interp_mean = np.mean(dep_interp, axis=0)
    # Plot average dependency
    ax.plot(grid_linspace, dep_interp_mean, linewidth=2, color='k')
    # Set x-axis limits
    ax.set_xlim(grid_linspace[0]-(grid_linspace[-1]-grid_linspace[0])/20,
                grid_linspace[-1]+(grid_linspace[-1]-grid_linspace[0])/20)
    # Set y-axis limits
    ax.set_ylim(
        np.min(dep_interp)-((np.max(dep_interp)-np.min(dep_interp))/20),
        np.max(dep_interp)+((np.max(dep_interp)-np.min(dep_interp))/20))
    # Compute influence
    influence = ((np.max(dep_interp_mean)-np.min(dep_interp_mean)) /
                 (y_range[1]-y_range[0]))
    # Make title string
    if class_label == -99:
        title_str = (task['predictor_name']+' | target: '+task['y_name'][0] +
                     ' dependence on'+'\n'+'predictors: '+feat+'\n' +
                     'Influence (% of '+task['y_name'][0]+' range) ' +
                     '{:.1f}%'.format(influence*100))
    elif class_label > -99:
        title_str = (task['predictor_name']+' | target: '+task['y_name'][0] +
                     ' | class: '+str(class_label)+' dependence on'+'\n' +
                     'predictors: '+feat+'\n' +
                     'Influence (% of class: '+str(class_label) +
                     ' probability) '+'{:.1f}%'.format(influence*100))
    # Set title
    ax.set_title(title_str, fontsize=10)
    # Set xlabel
    ax.set_xlabel('Predictor: '+feat, fontsize=10)
    # Make ylabel string
    if class_label == -99:
        ylabel_str = ('Target: '+task['y_name'][0])
    elif class_label > -99:
        ylabel_str = ('Probability for ' +
                      task['y_name'][0]+' | class: '+str(class_label))
    # Set ylabel
    ax.set_ylabel(ylabel_str, fontsize=10)
    # Make save path
    if class_label == -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_' +
                     task['y_name'][0]+'_on_'+feat)[:140]
    elif class_label > -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_' +
                     task['y_name'][0]+'_class_'+str(class_label)+'_on_' +
                     feat)[:140]
    # Save figure
    plt.savefig(save_path+'.png',
                dpi=150,
                bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg',
                bbox_inches='tight')
    # Show figure
    plt.show()
    # Save grid dataframe
    pd.DataFrame(grid).to_csv(save_path+'_x.csv')
    # Save feature dependence dataframe
    pd.DataFrame(dep).to_csv(save_path+'_y.csv')


def print_all_partial_dependences(task, results, plots_path):
    """
    Plot all partial dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate(results['true_values']))
    # Get range of true y values
    y_range = (np.min(unique_classes), np.max(unique_classes))
    # Make sub dir for partial dependencies
    plots_path = plots_path+'/'+'partial_dependencies'
    # Create plots sub dir
    create_dir(plots_path)
    # Iterate over predictors
    for i_feat, feat in enumerate(task['x_names']):
        # Get dependence data of current feature
        c_dep = [i[0][i_feat] for i in results['dependences']]
        # Get grid data of current feature
        c_grid = [i[1][i_feat] for i in results['dependences']]
        # chose depending on kind of problem
        if task['kind'] == 'clf' and len(unique_classes) == 2:
            # Print dependence
            try:
                print_partial_dependence(task, c_grid, c_dep, (0, 1), feat,
                                         plots_path)
            except ValueError:
                print('ValueError: plot skipped')
        elif task['kind'] == 'clf' and len(unique_classes) > 2:
            # Loop over classes
            for i_class, c_class in enumerate(unique_classes):
                # Get dependency data of current class
                dep = [i[i_class, :] for i in c_dep]
                # Print dependence
                try:
                    print_partial_dependence(task, c_grid, dep, (0, 1), feat,
                                             plots_path, class_label=c_class)
                except ValueError:
                    print('ValueError: plot skipped')
        else:
            # Print dependence
            try:
                print_partial_dependence(task, c_grid, c_dep, y_range, feat,
                                         plots_path)
            except ValueError:
                print('ValueError: plot skipped')


def print_inter_partial_dependence(task, grid, dep, y_range, features,
                                   plots_path, class_label=-99):
    """
    Plot single partial dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    grid : list
        List holding a grid per cross-validation repetition.
    dep : list
        List holding dependencies per cross-validation repetition.
    features : list string
        List of strings of the current features.
    plots_path : string
        Path to the plots.
    class_label : integer, optional
        Seperate model based feature importances by given class label.
        The default is -99.

    Returns
    -------
    None.

    """
    # Get grids axis a0
    grids_a0 = [i[0] for i in grid]
    # Get grids axis a1
    grids_a1 = [i[1] for i in grid]
    # Make common grid axis a0
    grid_a0 = np.linspace(np.min(np.hstack(grids_a0)),
                          np.max(np.hstack(grids_a0)), 20)
    # Make common grid axis a1
    grid_a1 = np.linspace(np.min(np.hstack(grids_a1)),
                          np.max(np.hstack(grids_a1)), 20)
    # Initialize interpolated preps
    deps_interp = []
    # Loop over repetitions of cv
    for i_rep, c_rep in enumerate(dep):
        # If more than 1
        if len(c_rep.shape) > 1:
            # Initialize a0 interpolated dependencies
            dep_a0_interp = np.empty((20, c_rep.shape[1]))
            # Loop over rows and interpolate rows to common grid
            for i_col, c_col in enumerate(c_rep.T):
                # Initiate interpolate
                inter = interp1d(grid[i_rep][0], c_col, kind='linear',
                                 axis=- 1, copy=True, bounds_error=None,
                                 fill_value='extrapolate', assume_sorted=True)
                # Interpolate rows to common grid
                dep_a0_interp[:, i_col] = inter(grid_a0)
            # Initialize a01 interpolated dependencies
            dep_a01_interp = np.empty((20, 20))
            # Loop over columns and Interpolate columns to common grid
            for i_row, c_row in enumerate(dep_a0_interp):
                # Initiate interpolate
                inter = interp1d(grid[i_rep][1], c_row, kind='linear',
                                 axis=- 1, copy=True, bounds_error=None,
                                 fill_value='extrapolate', assume_sorted=True)
                # Interpolate rows to common grid
                dep_a01_interp[i_row, :] = inter(grid_a1)
            # Append interpolated dependencies
            deps_interp.append(dep_a01_interp)
    # Make a mesh from common grid
    mesh = np.meshgrid(grid_a1, grid_a0)
    # Make average dependencies
    Z = np.mean(np.array(deps_interp), axis=0)
    # Plot filled contour
    plt.contourf(mesh[0], mesh[1], Z, levels=9, cmap='coolwarm')
    # Add colorbar
    plt.colorbar()
    # Add contour
    contours = plt.contour(mesh[0], mesh[1], Z, levels=9, linewidths=0.5,
                           colors='k')
    # Compute influence
    influence = (np.max(Z)-np.min(Z))/(y_range[1]-y_range[0])
    # Make title string
    if class_label == -99:
        title_str = (task['predictor_name']+' | target: '+task['y_name'][0] +
                     ' dependence on'+'\n'+'predictors: ' +
                     ' & '.join(features)+'\n'+'Influence (% of ' +
                     task['y_name'][0]+' range) ' +
                     '{:.1f}%'.format(influence*100))
    elif class_label > -99:
        title_str = (task['predictor_name']+' | target: '+task['y_name'][0] +
                     ' | class: '+str(class_label)+' dependence on '+'\n' +
                     'predictors: ' + ' & '.join(features)+'\n' +
                     'Influence (% of class: '+str(class_label) +
                     ' probability) '+'{:.1f}%'.format(influence*100))
    # Set title
    plt.title(title_str, fontsize=10)
    # Add x-label
    plt.xlabel('Predictor: '+features[1], fontsize=10)
    # Add y-label
    plt.ylabel('Predictor: '+features[0], fontsize=10)
    # Make save path
    if class_label == -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_' +
                     task['y_name'][0]+'_on_'+'_'.join(features))[:140]
    elif class_label > -99:
        save_path = (plots_path+'/'+task['predictor_name']+'_' +
                     task['y_name'][0]+'_class_'+str(class_label)+'_on_' +
                     '_'.join(features))[:140]
    # Add contour labels
    plt.clabel(contours, inline=True, fontsize=8)
    # Save figure
    plt.savefig(save_path+'.png',
                dpi=150,
                bbox_inches='tight')
    # Save figure
    plt.savefig(save_path+'.svg',
                bbox_inches='tight')
    # Show plot
    plt.show()


def print_all_inter_partial_dependences(task, results, plots_path):
    """
    Plot all ineraction partial dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.

    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate(results['true_values']))
    # Get range of true y values
    y_range = (np.min(unique_classes), np.max(unique_classes))
    # Make sub dir for partial dependencies
    plots_path = plots_path+'/'+'partial_dependencies_interaction'
    # Create plots sub dir
    create_dir(plots_path)
    # Get indices of predictors
    x_ind = list(range(len(task['x_names'])))
    # Get indices for interactions
    x_ind_inter = list(combinations(x_ind, 2))
    # Iterate over predictors
    for i_feat, feat in enumerate(x_ind_inter):
        # Get feature names
        features = [task['x_names'][feat[0]],
                    task['x_names'][feat[1]]]
        # Get dependence data of current feature
        c_dep = [i[2][i_feat] for i in results['dependences']]
        # Get grid data of current feature
        c_grid = [i[3][i_feat] for i in results['dependences']]
        # chose depending on kind of problem
        if task['kind'] == 'clf' and len(unique_classes) == 2:
            # Print dependence
            try:
                print_inter_partial_dependence(task, c_grid, c_dep, (0, 1),
                                               features, plots_path)
            except TypeError:
                print('TypeError: plot skipped')
        elif task['kind'] == 'clf' and len(unique_classes) > 2:
            # Loop over classes
            for i_class, c_class in enumerate(unique_classes):
                # Get dependency data of current class
                dep = [np.squeeze(i[i_class, :, :]) for i in c_dep]
                # Print dependence
                try:
                    print_inter_partial_dependence(task, c_grid, dep, (0, 1),
                                                   features, plots_path,
                                                   class_label=c_class)
                except TypeError:
                    print('TypeError: plot skipped')
        else:
            # Print dependence
            try:
                print_inter_partial_dependence(task, c_grid, c_dep, y_range,
                                               features, plots_path)
            except TypeError:
                print('TypeError: plot skipped')


def main():
    """
    Main function of plot results of machine-learning based data analysis.

    Returns
    -------
    None.

    """
    # Set initials ------------------------------------------------------------
    # Set seaborn plot style
    sns.set_style('darkgrid')
    # Loop over results dirs --------------------------------------------------

    # Get results subdirs of current directory
    res_dirs = [f.name for f in os.scandir('.') if
                f.is_dir() and f.name.startswith('res_')]
    # Loop over result dirs
    for res_dir in res_dirs:

        # Loop over tasks -----------------------------------------------------
        # Get task paths of current results subdir
        task_paths = [f.name for f in os.scandir('./'+str(res_dir)+'/')
                      if f.name.startswith('task_')]
        # Get results paths of current results subdir
        results_paths = [f.name for f in os.scandir('./'+str(res_dir)+'/')
                         if f.name.startswith('results_')]
        # Loop over tasks
        for i, task_path in enumerate(task_paths):

            # Load task and results -------------------------------------------
            # Load task description
            task = lfp(res_dir+'/'+task_path)
            # Load results
            results = lfp(res_dir+'/'+results_paths[i])

            # Create plots directory ------------------------------------------
            # Plots path
            plots_path = res_dir+'/plots_'+'_'.join(task['y_name'])
            # Create plots dir
            create_dir(plots_path)

            # Model fit -------------------------------------------------------
            if task['kind'] == 'reg':
                # Print model fit as scatter plot
                print_fit_regression_scatter(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_fit_regression_violin(task, results, plots_path)
            elif task['kind'] == 'reg_multi':
                print('not yet implemented')
            elif task['kind'] == 'clf':
                # Print model fit as confusion matrix
                print_fit_classification_confusion(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_fit_classification_violin(task, results, plots_path)

            # Model based predictors importance -------------------------------
            # Print model based importance
            print_all_model_importances(task, results, plots_path)

            # Permutation based predictors importance -------------------------
            # Print permutation based importance
            print_permutation_importance(task, results, plots_path)

            # SHAP based predictors importance --------------------------------
            # Print permutation based importance
            print_all_shap_importances(task, results, plots_path)

            # SHAP values -----------------------------------------------------
            # Print permutation based importance
            print_all_shap_values(task, results, plots_path)

            # Partial dependence ----------------------------------------------
            # Print partial dependence of dep variables to ind variables
            print_all_partial_dependences(task, results, plots_path)

            # Partial dependence ----------------------------------------------
            # Print interaction partial dependence of dep variables to ind var
            print_all_inter_partial_dependences(task, results, plots_path)


if __name__ == "__main__":
    main()
