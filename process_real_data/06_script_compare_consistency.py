import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj
import tools.graph_visu as gv
import tools.graph_processing as gp
import matplotlib.pyplot as plt
import pickle as p

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    path_to_figs = '../data/Oasis_original_new_with_dummy/figures'
    reg_or_unreg = ''#'_unreg'#''
    methods = ['MatchEig','media','media_no_excl','media_no_excl_neg_values','neuroimage', 'kerGM', 'mALS', 'mSync', 'CAO']#, 'kmeans_70_real_data_dummy','kmeans_90_real_data_dummy','kmeans_110_real_data_dummy']
    #, 'mSync', 'CAO'

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    nb_bins = 20
    bins = np.arange(0,1,0.05)
    dens = False
    fig1, ax = plt.subplots(1, len(methods), sharey=True, sharex=True)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)

        pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_"+ method + reg_or_unreg +".pck"),"rb")
        nodeCstPerGraph = pickle.load(pickle_in)
        pickle_in.close()

        print("Average across all nodes of node consistency "+method + reg_or_unreg +":", np.mean(nodeCstPerGraph), np.std(nodeCstPerGraph))


        #print(np.mean(nodeCstPerGraph,1))
        #print(np.std(nodeCstPerGraph,1))
    #print(np.mean(nodeCstPerGraph_mSync,1))
    #print(np.mean(nodeCstPerGraph_KerGM,1))
    #print(np.mean(nodeCstPerGraph_CAO,1))
    #rank_mSync = np.linalg.matrix_rank(matching_mSync)
    #print(rank_mSync)


        ax[ind].hist(nodeCstPerGraph.flatten(), density=dens, bins=bins)  # density=False would make counts
        ax[ind].set_ylabel('Frequency')
        ax[ind].set_xlabel('consistency')
        ax[ind].set_title(method)
        ax[ind].grid(True)



    plt.show()
