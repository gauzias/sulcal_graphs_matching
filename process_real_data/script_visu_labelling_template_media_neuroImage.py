import sys
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import os
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy


if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    label_attribute = 'label_neuroimage'#'label_media'

    #path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_labelled_pits_graphs'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/labelled_pits_graphs_coords'
    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    print(len(list_graphs))
    for g in list_graphs:
        nx.set_node_attributes(g, values=False, name="is_dummy")
    # [-2, 0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33,
    #  34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    #  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
    # 89
    # path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    # list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # print(len(list_graphs)) #137
    #
    # for i, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
    #[-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
    # 94
    # pick_corr = open('/mnt/data/work/python_sandBox/Graph_matching/data/graph_correspondence_new.pickle', "rb")
    # corr_dict = p.load(pick_corr)
    # pick_corr.close()


    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)

    print('create_clusters_lists')
    print(list_graphs[0].nodes[0])
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    labels = list(cluster_dict.keys())
    labels.sort()
    print(labels)
    print(len(labels))
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)


    vb_sc = gv.visbrain_plot(reg_mesh)
    vmin=0
    vmax=92#329#92
    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        labels_subj = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels_subj])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None, c_map='nipy_spectral',  vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)
    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                        nodes_size=60, nodes_mask=None, c_map='nipy_spectral', symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()






    vb_sc2 = gv.visbrain_plot(reg_mesh)

    label_to_plot = 2#222
    for ind,g in enumerate(list_graphs):

        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        color_label_to_plot = np.ones(color_label.shape)
        color_label_to_plot[color_label == label_to_plot]=0
        #print(color_label)
        if np.sum(color_label == label_to_plot)==0:
            print(ind)
        else:
            s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_to_plot, nodes_mask=None, c_map='nipy_spectral',  vmin=0, vmax=1)
            vb_sc2.add_to_subplot(s_obj)
    vb_sc2.preview()
