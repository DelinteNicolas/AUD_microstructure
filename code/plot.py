# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:38:01 2023

@author: N. Delinte
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_ind
from statannotations.Annotator import Annotator
from unravel.viz import plot_metric_along_trajectory, plot_trk
from unravel.stream import extract_nodes, get_roi_sections_from_nodes


def jsonToPandas(jsonFilePath: str, control_list_path: str = None,
                 unwanted_list_path: str = None, fullname: bool = True):

    with open(jsonFilePath) as file:
        dic = json.load(file)['Mean']

    if unwanted_list_path:
        with open(unwanted_list_path) as file:
            unwanted_list = json.load(file)
        dic = {k: v for k, v in dic.items() if k not in unwanted_list}

    reform = {(level1_key, level2_key, level3_key): values
              for level1_key, level2_dict in dic.items()
              for level2_key, level3_dict in level2_dict.items()
              for level3_key, values in level3_dict.items()}

    p = pd.DataFrame(reform, index=['Value']).T
    p = p.rename_axis(['Patient', 'Region', 'Metric'])

    if control_list_path is not None:

        with open(control_list_path) as file:
            control_list = json.load(file)

        p_list = p.index.get_level_values(0)
        if fullname:
            pat = [name in control_list for name in p_list]
        else:
            pat = [name[:-3] in control_list for name in p_list]
        p['Patient type'] = ['control' if p else 'case' for p in pat]

    t_list = p.index.get_level_values(0)
    pat = [n[-2:] for n in t_list]
    p['Time'] = pat

    return p


def plot_df(df, region: str, metric: str, show_E3: bool = False,
            adjusted_value: bool = True):

    df = df.loc[:, region, metric]

    if not show_E3:
        df = df[df['Time'] != 'E3']

    y = 'Value_adj' if adjusted_value else 'Value'

    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.violinplot(data=df, x='Time', y=y, cut=0, hue='Patient type',
                        palette=sns.color_palette('pastel'),
                        order=['E1', 'E2'], hue_order=['case', 'control'])
    ax.set_ylabel(metric)
    ax.set_title(metric + ' values of all patient for the ' + region)

    pairs = [(("E1", "case"), ("E1", "control")),
             (("E2", "case"), ("E2", "control")),
             (("E1", "case"), ("E2", "case")),
             (("E1", "control"), ("E2", "control"))]
    if show_E3:
        pairs += [(("E2", "case"), ("E3", "case")),
                  (("E1", "case"), ("E3", "case"))]

    annotator = Annotator(ax, pairs, data=df, x='Time', y=y,
                          hue='Patient type',
                          order=['E1', 'E2'], hue_order=['case', 'control'])
    annotator.configure(test="t-test_welch").apply_and_annotate()


def get_mean_trajectories(dic: dict, region: str, metric: str,
                          control_list: list, time: str = ''):

    r = region
    m = metric
    pat_1 = True
    con_1 = True
    for p in dic.keys():
        if time in p:
            try:
                if p in control_list:
                    if con_1:
                        controls = np.array(dic[p][r][m])[..., np.newaxis]
                        con_1 = False
                    else:
                        controls = np.append(controls,
                                             np.array(
                                                 dic[p][r][m])[..., np.newaxis],
                                             axis=1)
                else:
                    if pat_1:
                        patients = np.array(dic[p][r][m])[..., np.newaxis]
                        pat_1 = False
                    else:
                        patients = np.append(patients,
                                             np.array(
                                                 dic[p][r][m])[..., np.newaxis],
                                             axis=1)
            except KeyError:
                print(p)
                continue
    return patients, controls


def linear_regression(df, correction_metric: str):

    for r in list(df.index.unique(level=1)):
        x = np.array(df.loc[:, r, correction_metric]['Value'])
        for m in list(df.index.unique(level=2)):
            y = np.array(df.loc[:, r, m]['Value'])
            res = linregress(x, y)

            for patient in list(df.index.unique(level=0)):
                try:
                    a = (np.mean(y) + df.loc[(patient, r, m), 'Value']
                         - (res.intercept + res.slope
                            * np.array(df.loc[patient, r, correction_metric]
                                       ['Value'])))
                    df.loc[(patient, r, m), 'Value_adj'] = a
                except KeyError:
                    continue
    return df


if __name__ == '__main__':

    root = '../data/'

    dictionnary_path = root + 'unravel_mean_ang_tsl_clean.json'
    trajectory_dictionnary_path = root + 'unravel_trajectory_ang_tsl_clean.json'
    unwanted_list_path = root + 'unwanted.json'
    control_list_path = root + 'control_list.json'

    correction_metric = 'snr'
    plot_regress = True
    show_E3 = False
    plot_violins = True

    trajectory_region = 'uf_right'
    trajectory_metric = 'AD'

    flip = True

    trk_file = root + 'sub02_E1_' + trajectory_region+'.trk'

    # Violin plots -------------------------------------------------------------

    df = jsonToPandas(dictionnary_path, control_list_path, unwanted_list_path)
    df['Value_adj'] = df['Value']
    df = linear_regression(df, correction_metric)

    if plot_regress:

        plt.figure()
        x = df.loc[:, trajectory_region, 'snr']['Value']
        y = df.loc[:, trajectory_region, 'FA']['Value']
        res = linregress(x, y)
        plt.plot(x, y, 'o', label='original data')
        plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
        y_adj = np.mean(y)+y-(res.intercept+res.slope*x)
        plt.scatter(x, y_adj, c='orange', label='regressed data')
        plt.legend()
        plt.show()

    region_list = list(df.index.get_level_values(1).unique())
    metric_list = list(df.index.get_level_values(2).unique())

    # metric_list.remove('FA_DTI')
    # metric_list.remove('AD_DTI')
    # metric_list.remove('RD_DTI')
    # metric_list.remove('MD_DTI')
    # metric_list.remove('wFA')
    # metric_list.remove('wAD')
    # metric_list.remove('wRD')
    # metric_list.remove('wMD')

    # metric_list.remove('fvf')
    # metric_list.remove('fvf_tot')
    # metric_list.remove('frac')
    # metric_list.remove('stream_count')
    # metric_list.remove('voxel_count')

    metric_list.remove('snr')

    # metric_list.remove('movement')
    # metric_list.remove('frac_csf')

    # region_list.remove('slf_right')
    # region_list.remove('slf_left')
    # region_list.remove('ci_right')
    # region_list.remove('ci_left')

    # unwanted_metrics = ['snr']

    # for m in unwanted_metrics:
    #     metric_list.remove(m)

    # unwanted_regions = []

    # for r in unwanted_regions:
    #     region_list.remove(r)

    if plot_violins:
        for r in region_list:
            for m in metric_list:
                plot_df(df, r, m, show_E3=show_E3)

    pairs = [(("E1", "case"), ("E1", "control")),
             (("E2", "case"), ("E2", "control")),
             (("E1", "case"), ("E2", "case")),
             (("E1", "control"), ("E2", "control"))]
    if show_E3:
        pairs += [(("E2", "case"), ("E3", "case")),
                  (("E1", "case"), ("E3", "case"))]

    df = df.set_index('Time', append=True)
    df = df.set_index('Patient type', append=True)

    pvals, pr, pm, pp = [], [], [], []

    for r in region_list:
        for m in metric_list:
            for p in pairs:
                a = df.loc[:, r, m, p[0][0], p[0][1]]['Value_adj']
                b = df.loc[:, r, m, p[1][0], p[1][1]]['Value_adj']
                # Welch t-test
                _, pval = ttest_ind(a, b, equal_var=False)
                if pval < 0.05:
                    pvals.append(pval)
                    pr.append(r)
                    pm.append(m)
                    pp.append(p)

    psorted = np.sort(np.array(pvals))
    idx_s = np.argsort(np.array(pvals))

    # Bonferroni

    # number of comparisons = metrics * rois * time
    m = len(metric_list)*len(region_list)*4
    pb = 0.05 / m
    significant = np.sum(psorted < pb)
    for i in range(significant):
        idx = idx_s[i]
        print(pr[idx], pm[idx], pp[idx], pvals[idx])

    # Along metric analysis ----------------------------------------------------

    with open(trajectory_dictionnary_path, 'r') as file:
        data_t = json.load(file)['Mean']

    with open(unwanted_list_path) as file:
        unwanted_list = json.load(file)

    with open(control_list_path) as file:
        control_list = json.load(file)

    for p in unwanted_list:
        try:
            del data_t[p]
        except KeyError:
            continue

    patients, controls = get_mean_trajectories(data_t, trajectory_region,
                                               trajectory_metric,
                                               control_list, time='E1')

    plt.figure(figsize=(6, 5))

    mean_p = np.mean(patients, axis=1)
    std_p = np.std(patients, axis=1)
    mean_c = np.mean(controls, axis=1)
    std_c = np.std(controls, axis=1)

    l = patients.shape[0]

    if flip:
        mean_p[1:l] = mean_p[:-l:-1]
        std_p[1:l] = std_p[:-l:-1]
        mean_c[1:l] = mean_c[:-l:-1]
        std_c[1:l] = std_c[:-l:-1]

    plot_metric_along_trajectory(mean_p, std_p,
                                 new_fig=False, label='case')
    plot_metric_along_trajectory(mean_c, std_c,
                                 new_fig=False, label='control')

    plt.xlabel('Trajectory')
    plt.ylabel(trajectory_metric)
    # plt.ylim([0.001, 0.0025])
    # plt.ylim([0.001, 0.0020])
    # plt.ylim([0.5, 1])
    # plt.ylim([0.3, 0.7])
    plt.title('Evolution of '+trajectory_metric+' along the '+trajectory_region)
    plt.legend()

    # Showing tracts -----------------------------------------------------------

    point_array = extract_nodes(trk_file, nodes=8)
    roi = get_roi_sections_from_nodes(trk_file, point_array)

    plot_trk(trk_file, scalar=roi, background='white', color_map='Set3')
