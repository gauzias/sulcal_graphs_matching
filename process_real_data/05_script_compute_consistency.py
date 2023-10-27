import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import os
import pickle
import scipy.io as sco
import tools.graph_processing as gp
import tools.clusters_analysis as gca


if __name__ == "__main__":
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_X = '../data/Oasis_original_new_with_dummy'
    reg_or_unreg = ''#'_unreg'#''
    methods =  ['MatchEig'] #['media_no_excl']#['neuroimage'], #, 'mALS', 'kmeans_70_real_data', 'mSync']#,'kerGM', 'CAO']#['media', 'neuroimage']#,
    #methods = ['kmeans_70_real_data_dummy','kmeans_90_real_data_dummy','kmeans_110_real_data_dummy']

    # load the graphs
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    nb_graphs = len(list_graphs)
    print('nb graphs ', nb_graphs)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)
        # load the assignment matrix
        if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + "_dummy.mat")
        else:
            file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
        if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']

        # get the associated number of nodes
        nb_nodes = int(X.shape[0]/nb_graphs)
        print(nb_nodes)
        nodeCstPerGraph = gca.compute_node_consistency(X, nb_graphs, nb_nodes)
        pickle_out = open(os.path.join(path_to_consistency, "nodeCstPerGraph_"+method+reg_or_unreg+".pck"), "wb")
        pickle.dump(nodeCstPerGraph, pickle_out)
        pickle_out.close()

    # # read the assignment matrices
    # x_mSync = sco.loadmat(os.path.join(path_to_match_mat, "X_mSync"+reg_or_unreg+".mat"))["X"]
    # x_mALS = sco.loadmat(os.path.join(path_to_match_mat, "X_mALS"+reg_or_unreg+".mat"))["X"]
    # x_cao = sco.loadmat(os.path.join(path_to_match_mat, "X_cao_cst_o.mat"))["X"]
    # x_kerGM = sco.loadmat(os.path.join(path_to_match_mat,"X_pairwise_kergm"+reg_or_unreg+".mat"))["full_assignment_mat"]
    #
    # # get the associated number of nodes
    # nb_nodes = int(x_mALS.shape[0]/nb_graphs)
    # print(nb_nodes)
    # nodeCstPerGraph_mALS = gca.compute_node_consistency(x_mALS, nb_graphs, nb_nodes)
    # pickle_out = open(os.path.join(path_to_consistency, "nodeCstPerGraph_mALS"+reg_or_unreg+".pck"), "wb")
    # pickle.dump(nodeCstPerGraph_mALS, pickle_out)
    # pickle_out.close()

    # nb_nodes = int(x_mSync.shape[0]/nb_graphs)
    # print(nb_nodes)
    # nodeCstPerGraph_mSync = gca.compute_node_consistency(x_mSync, nb_graphs, nb_nodes)
    # pickle_out = open(os.path.join(path_to_consistency, "nodeCstPerGraph_mSync"+reg_or_unreg+".pck"),"wb")
    # pickle.dump(nodeCstPerGraph_mSync, pickle_out)
    # pickle_out.close()

    # nb_nodes = int(x_kerGM.shape[0]/nb_graphs)
    # nodeCstPerGraph_KerGM = gca.compute_node_consistency(x_kerGM, nb_graphs, nb_nodes)
    # pickle_out = open(os.path.join(path_to_consistency, "nodeCstPerGraph_KerGM"+reg_or_unreg+".pck"),"wb")
    # pickle.dump(nodeCstPerGraph_KerGM, pickle_out)
    # pickle_out.close()

    # nb_nodes = int(x_cao.shape[0]/nb_graphs)
    # nodeCstPerGraph_CAO = gca.compute_node_consistency(x_cao, nb_graphs, nb_nodes)
    # pickle_out = open(os.path.join(path_to_consistency, "nodeCstPerGraph_CAO.pck"),"wb")
    # pickle.dump(nodeCstPerGraph_CAO, pickle_out)
    # pickle_out.close()

    # nb_nodes = int(matching_Hippi.shape[0]/nb_graphs)
    # #print(nb_node)
    # nodeCstPerGraph_Hippi = compute_node_consistency(matching_Hippi, nb_graphs, nb_nodes)
    # pickle_out = open(os.path.join(path_to_read,"nodeCstPerGraph_Hippi.pck"),"wb")
    # pickle.dump(nodeCstPerGraph_Hippi, pickle_out)
    # pickle_out.close()


