import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching")

import os
import numpy as np
import xlrd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import tools.graph_processing as pg
import networkx as nx
#import pitsgraph as pg

if __name__ == "__main__":
    output_dir = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/ABIDE/graph_lh'
    hemi = 'lh'#, 'rh'

    # try to load subjects graph data
    graph_files_list = list()
    dir_subj_files_list=os.listdir(output_dir)
    for fil in dir_subj_files_list:
        if hemi in fil:
            if fil.find('.gpickle'):
                graph_files_list.append(fil)
    print(len(graph_files_list)) # 140

    processed_subjects = list()
    pitgraphs_list = list()
    subjects_list = list()
    for graph_file in graph_files_list:
        print('------' + graph_file)
        print(graph_file[:7])
        g = nx.read_gpickle(os.path.join(output_dir, graph_file))
        pitgraphs_list.append(g)
        subjects_list.append(graph_file[:7])

    subjects_mean_depth = list()
    subjects_nb_pits = list()
    for g in pitgraphs_list:
        subjects_mean_depth.append(np.mean(pg.graph_nodes_attribute(g, 'depth')))
        subjects_nb_pits.append(len(g))
    # load phenotype data
    subjects_info = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/ABIDE/Phenotypic_V1_0b_traitements_visual_check_GA.xls'
    #subjects_info = '/mnt/data/work/python_sandBox/pits_graph/data_pits_graph/ABIDE/Phenotypic_V1_0b_traitements_visual_check_GA.xls'
    # Create a update subjects list
    wb = xlrd.open_workbook(subjects_info)
    sh = wb.sheet_by_name(u'Feuille1')

    # full_diag = sh.col_values(3)[1:]
    subjects_list_table = list()
    QC_table = list()
    diag_table = list()
    centers_table = list()
    sex_table = list()
    age_table = list()
    handedness_table = list()
    phenotype_colnames = [sh.row(0)[0], sh.row(0)[2], sh.row(0)[7], sh.row(0)[8], sh.row(0)[9]]
    print(phenotype_colnames)
    for ind in range(1, sh.nrows):
        #if sh.col_values(2)[ind] < 3.0:
        subjects_list_table.append('{:07.0f}'.format(sh.col_values(1)[ind]))
        QC_table.append(float(sh.col_values(2)[ind]))
        centers_table.append(str(sh.col_values(0)[ind]))
        age_table.append(float(sh.col_values(7)[ind]))
        sex_table.append(str(sh.col_values(8)[ind]))
        handedness_table.append(str(sh.col_values(9)[ind]))
        if sh.col_values(5)[ind] == 1:
            diag_table.append('asd')
        else:
            diag_table.append('ctrl')
    QC_table = np.array(QC_table)
    diag_table = np.array(diag_table)
    centers_table = np.array(centers_table)
    sex_table = np.array(sex_table)
    age_table = np.array(age_table)
    handedness_table = np.array(handedness_table)

    print('nb subjects in table = '+str(len(subjects_list_table)))
    print(subjects_list_table)
    # keep phenotype only for subjects with data loaded
    inds_keep = list()
    for s in subjects_list:
        if s in subjects_list_table:
            inds_keep.append(subjects_list_table.index(s))
    inds_keep = np.array(inds_keep, int)
    QC = QC_table[inds_keep]
    diag = diag_table[inds_keep]
    age = age_table[inds_keep]
    centers = centers_table[inds_keep]
    sex = sex_table[inds_keep]
    handedness = handedness_table[inds_keep]

    # check correspondance
    subjects_list_check = np.array(subjects_list_table)[inds_keep]
    print(len(set(subjects_list_check).intersection(set(subjects_list))))

    # joblib.dump(pitgraphs_dict, pitgraphs_path, compress=3)
    # joblib.dump([subjects_list, subjects_nb_pits, subjects_mean_depth], subjects_list_file, compress=3)
    # joblib.dump([diag, age, centers, sex, handedness], subjects_phenotype_file, compress=3)

    # general data plot
    # [diag, age, centers, sex, handedness] = joblib.load(subjects_phenotype_file)
    asd_inds = diag =='asd'
    asd_nb_pits = np.array(subjects_nb_pits)[asd_inds]
    asd_mean_depth = np.array(subjects_mean_depth)[asd_inds]
    ctrl_inds = diag =='ctrl'
    ctrl_nb_pits = np.array(subjects_nb_pits)[ctrl_inds]
    ctrl_mean_depth = np.array(subjects_mean_depth)[ctrl_inds]
    hist_cent_nb_pits = list()
    hist_cent_mean_depth = list()
    leg = list()
    for ce in set(centers):
        inds_ce = centers ==ce
        hist_cent_nb_pits.append(np.array(subjects_nb_pits)[inds_ce])
        hist_cent_mean_depth.append(np.array(subjects_mean_depth)[inds_ce])
        leg.append(str(np.sum(inds_ce) ) +'  ' +ce)
        print(str(np.sum(inds_ce) ) +'  ' +ce +' mean age =  ' +str(np.mean(age[inds_ce])))

    fig, axes = plt.subplots(3, 2)
    axes[0, 0].hist(subjects_nb_pits)
    axes[0, 0].set_title('nb pits all subjects')
    axes[0, 1].hist(subjects_mean_depth)
    axes[0, 1].set_title('mean pits depth all subjects')
    axes[1, 0].hist([asd_nb_pits ,ctrl_nb_pits], label=[str(np.sum(asd_inds) ) +' asd' ,str(np.sum(ctrl_inds) ) +' crtl'], density=True)
    axes[1, 0].legend()
    axes[1, 0].set_title('nb pits by diagnostic')
    axes[1, 1].hist([asd_mean_depth ,ctrl_mean_depth], label=[str(np.sum(asd_inds) ) +' asd' ,str(np.sum(ctrl_inds) ) +' crtl'], density=True)
    axes[1, 1].legend()
    axes[1, 1].set_title('mean pits depth by diagnostic')
    axes[2, 0].hist(hist_cent_nb_pits, density=True, label=leg)
    axes[2, 0].legend()
    axes[2, 0].set_title('nb pits by center')
    axes[2, 1].hist(hist_cent_mean_depth, density=True, label=leg)
    # axes[2,1].legend()
    axes[2, 1].set_title('mean pits depth by center')
    plt.show()

    # f = pitTls.inline_plot_graphs(pitgraphs_dict, subjects_list, np.array(subjects_nb_pits), sample=20)
    # plt.show()