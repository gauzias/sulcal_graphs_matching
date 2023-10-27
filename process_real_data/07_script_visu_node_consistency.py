import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj
import tools.graph_visu as gv
import tools.graph_processing as gp
import scipy.io as sco
import tools.clusters_analysis as gca

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'  # lh.OASIS_testGrp_average_inflated.gii'
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    c_map = 'Reds_r'#'Greens'#'hot'#'brg'#''nipy_spectral'
    path_to_X = "../data/Oasis_original_new_with_dummy"
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    reg_or_unreg = ''  # '_unreg'#''
    method = 'kerGM'#'mALS'#'neuroimage'#'MatchEig'#'kerGM'#'CAO'#'media_no_excl'#'kerGM'#'mSync'#'media_no_excl'#'mSync'#  # #'mALS'#'kmeans_70_real_data_dummy'#'media'#'CAO'#'mALS'#
    default_label = -0.1
    vmin = 0#0.5
    vmax = 0.5#1


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
    #
    # largest_ind=22
    # label_attribute = 'labelling_from_assgn'
    # # load the assignment matrix
    # if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
    #     file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + "_dummy.mat")
    # else:
    #     file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
    # if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
    #     X = sco.loadmat(file_X)["full_assignment_mat"]
    # else:
    #     X = sco.loadmat(file_X)['X']
    #
    # trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute, default_label_value=default_label)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    print(cluster_dict.keys())
#    cluster_dict2 = gca.create_clusters_lists(list_graphs, label_attribute='label_neuroimage')

    print(len(cluster_dict))
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    # load consistency
    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_"+ method + reg_or_unreg +".pck"),"rb")
    nodeCstPerGraph = pickle.load(pickle_in)
    pickle_in.close()

    clusters_cst = gca.get_consistency_per_cluster(cluster_dict, nodeCstPerGraph)
    print(np.min(clusters_cst), np.max(clusters_cst))
    print(np.mean(clusters_cst), np.std(clusters_cst))
    print(clusters_cst)

    # data shown at the graph nodes level is the average across ref_graphs
    data_node_cstr = np.mean(nodeCstPerGraph, 1)

    vb_sc = gv.visbrain_plot(reg_mesh)#None
    for ind_g, g in enumerate(list_graphs):
        data_mask = gp.remove_dummy_nodes(g)
        data_node_cstr = nodeCstPerGraph[:,ind_g]
        #vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
                                                        nodes_size=None, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh, attribute_vertex_index='ico100_7_vertex_index')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clusters_cst,
                                                        nodes_size=90, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    #vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=60)

    vb_sc.preview()
    #vb_sc.screenshot(os.path.join(path_to_figs, 'consistency_'+method+'.png'))

#
#     #
#     list_graphs = gp.load_graphs_in_list(path_to_graphs)
#     vb_sc1 = gv.visbrain_plot(reg_mesh)
#     #clim=(0.8, 0.95)
#     g=list_graphs[ind_g]
#     data_mask = gp.remove_dummy_nodes(g)
#     data_node_cstr = np.mean(nodeCstPerGraph,1)
#     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
#     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
#                                                         nodes_size=60, nodes_mask=None, c_map='hot', symbol='disc',
#                                                         vmin=vmin, vmax=vmax)
#
#     vb_sc1.add_to_subplot(s_obj)
#     vb_sc1.preview()

    #vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=60)
    # Ry(180)
    # transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    # mesh.apply_transform(transfo_full)
    # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index'+graph_att, mesh)
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
    #                                                     nodes_size=60, nodes_mask=None, c_map='hot', symbol='disc',
    #                                                     vmin=vmin, vmax=vmax)
    #
    # vb_sc2 = gv.visbrain_plot(mesh)
    # vb_sc2.add_to_subplot(s_obj)
    # vb_sc2.preview()


    # vb_sc = None
    # clim=(0.8, 0.95)
    # s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    # vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    # visb_sc_shape = get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    #
    # # Ry(180)
    # transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    # mesh.apply_transform(transfo_full)
    # s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    # vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    # visb_sc_shape = get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=visb_sc_shape[1] + 1, width_max=200)
    # vb_sc.preview()
    #
    # print(np.min(np.mean(nodeCstPerGraph_Hippi, 1)))
    # print(np.max(np.mean(nodeCstPerGraph_Hippi, 1)))
