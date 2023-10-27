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
    c_map = 'nipy_spectral'
    path_to_X = "../data/Oasis_original_new_with_dummy"
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    reg_or_unreg = ''  # '_unreg'#''
    method = 'media_no_excl'#'neuroimage'#
    default_label = -0.1
    vmin = -0.1
    vmax = 1.1
#     vmin = 0
#     vmax = 93#300

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    print('----------------------------')
    print(method)
    largest_ind=22
    label_attribute = 'labelling_from_assgn'
    # load the assignment matrix
    if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
        file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + "_dummy.mat")
    else:
        file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
    if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
        X = sco.loadmat(file_X)["full_assignment_mat"]
    else:
        X = sco.loadmat(file_X)['X']

    trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute, default_label_value=default_label)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    #cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute='label_'+method)

    print(cluster_dict.keys())

    print(len(cluster_dict))
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    for g in list_graphs:
        gp.remove_dummy_nodes(g)

    vb_sc = gv.visbrain_plot(reg_mesh)#None

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        #labels = nx.get_node_attributes(g, 'label_media').values()
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None,
                                                        c_map=c_map,  vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh, attribute_vertex_index='ico100_7_vertex_index')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                        nodes_size=90, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()

