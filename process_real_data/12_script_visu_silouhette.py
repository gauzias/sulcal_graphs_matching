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
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    c_map = 'nipy_spectral'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_silhouette = '../data/Oasis_original_new_with_dummy/silhouette'
    path_to_figs = '../data/Oasis_original_new_with_dummy/figures'
    path_to_X = "../data/Oasis_original_new_with_dummy"
    reg_or_unreg = ''#'_unreg'#''
    method = 'MatchEig'#'CAO'#'mSync'#'neuroimage'#'mALS'#'media'#'kerGM'#'kerGM'#'CAO'#'kerGM'#'kmeans_70_real_data_dummy'#
    default_label = -0.1
    vmin = -1
    vmax = 1

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    print('----------------------------')
    print(method)
    # compute the labeling from the assignment matrix when needed
    if 'media' in method:
        label_attribute = 'label_media'
    elif 'neuroimage' in method:
        label_attribute = 'label_neuroimage'
    else:
        largest_ind=22
        label_attribute = 'labelling_from_assgn'
        # load the assignment matrix
        file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
        if 'kerGM' in method:
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']
        trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute, default_label_value=default_label)
        print(np.unique(trans_l))
    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    print(len(cluster_dict))
    print(cluster_dict.keys())
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    print('load precomputed silhouette value')
    pickle_in = open(os.path.join(path_to_silhouette, 'labelling_'+method+reg_or_unreg+'_silhouette.gpickle'), "rb")
    silhouette_dict = p.load(pickle_in)
    pickle_in.close()

    sil_keys = list(silhouette_dict.keys())
    sil_keys.sort()
    clus_keys = list(cluster_dict.keys())
    clus_keys.sort()
    diff = np.array(sil_keys)-np.array(clus_keys)
    clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)
    for i, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)

    vb_sc = gv.visbrain_plot(reg_mesh)
    for g_i, g in enumerate(list_graphs):
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        nodes_silhouette = gca.get_silhouette_per_graph(cluster_dict, silhouette_dict, g_i, len(g))
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=nodes_silhouette, nodes_mask=None,
                                                        c_map=c_map,  vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=3, width_max=200)
    vb_sc.preview()
    # Ry(180)
    # transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    # reg_mesh.apply_transform(transfo_full)
    # vb_sc1 = gv.visbrain_plot(reg_mesh)
    #
    # centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
    #                                                     nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
    #                                                     vmin=-1, vmax=1)
    #
    # vb_sc1.add_to_subplot(s_obj)
    # # vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1,
    # #                            col=3, width_max=200)
    # vb_sc1.preview()

    #vb_sc.screenshot(os.path.join(path_to_figs, 'silhouette_'+label_attribute+'.png'))
    print(np.mean(clust_silhouette))
    print(np.std(clust_silhouette))
    print(len(clust_silhouette))