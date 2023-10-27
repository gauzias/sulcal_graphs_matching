import os
import sys
#sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy
from visbrain.objects import SourceObj, ColorbarObj


if __name__ == "__main__":
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_silhouette = '../data/Oasis_original_new_with_dummy/silhouette'
    reg_or_unreg = '_unreg'#''
    path_to_mALS = "../data/Oasis_original_new_with_dummy/X_mALS"+reg_or_unreg+".mat"
    path_to_mSync = "../data/Oasis_original_new_with_dummy/X_mSync"+reg_or_unreg+".mat"
    path_to_CAO = "../data/Oasis_original_new_with_dummy/X_CAO.mat"
    path_to_kerGM = "../data/Oasis_original_new_with_dummy/X_pairwise_kergm"+reg_or_unreg+".mat"
    mesh = sio.load_mesh(template_mesh)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for i,g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        #print(i,len(g)) # allow to identify the largest graph (22)

    X_mALS = sco.loadmat(path_to_mALS)['X']
    X_mSync = sco.loadmat(path_to_mSync)['X']
    X_CAO = sco.loadmat(path_to_CAO)['X']
    X_kerGM = sco.loadmat(path_to_kerGM)["full_assignment_mat"]

    largest_ind = 22#24
    print('get_clusters_from_assignment')
    #label_attribute = 'labelling_hippi'
    #gca.get_clusters_from_assignment_hippi(list_graphs, X_Hippi, largest_ind, mesh, label_attribute)
    #label_attribute = 'labelling_CAO'
    #gca.get_clusters_from_assignment(list_graphs, X_CAO, largest_ind, mesh, label_attribute)
    label_attribute = 'labelling_kerGM'+reg_or_unreg
    gca.get_labelling_from_assignment(list_graphs, X_kerGM, largest_ind, mesh, label_attribute)
    # label_attribute = 'labelling_mSync'+reg_or_unreg
    # gca.get_labelling_from_assignment(list_graphs, X_mSync, largest_ind, mesh, label_attribute)
    # label_attribute = 'labelling_mALS'+reg_or_unreg
    # gca.get_labelling_from_assignment(list_graphs, X_mALS, largest_ind, mesh, label_attribute)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

    # Calculate or load the silhouette values
    # if path_silhouette != "":
    #     pickle_in = open(path_silhouette, "rb")
    #     silhouette_dict = pickle.load(pickle_in)
    #     pickle_in.close()
    #
    # else:
    print('get_all_silhouette_value')
    silhouette_dict = gca.get_all_silhouette_value(list_graphs, cluster_dict)
    pickle_out = open(os.path.join(path_to_silhouette, label_attribute+'_silhouette.gpickle'), "wb")
    p.dump(silhouette_dict, pickle_out)
    #pickle_out = open(os.path.join(path_to_silhouette, label_attribute+'_silhouette.gpickle'), "rb")
    #silhouette_dict = p.load(pickle_out)
    #pickle_out.close()
    clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)

    # # save the silhouette value if necessary
    # if path_to_save != "":
    #     pickle_out = open(path_to_save, "wb")
    #     pickle.dump(silhouette_dict, pickle_out)
    #     pickle_out.close()
    print('visu')
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)
    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                        vmin=-1, vmax=1)

    vb_sc.add_to_subplot(s_obj)
    # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=3, width_max=200)
    vb_sc.preview()
    #vb_sc.screenshot(os.path.join(path_to_silhouette, label_attribute+'.png'))
    print(np.mean(clust_silhouette))
    print(np.std(clust_silhouette))
    print(len(clust_silhouette))