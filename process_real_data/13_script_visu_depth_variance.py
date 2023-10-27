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
import matplotlib.pyplot as plt

def create_clusters_lists(list_graphs, label_attribute="label_dbscan"):
    """
    Given a list of graphs, return a list of list that represents the clusters.
    Each inside list represent one cluster and each elemnt of the cluster is
    a tuple (graph_number, node_in_the_graph).
    """

    result_dict = {}
    label_depths = {}

    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"]:
                label_cluster = graph.nodes[node][label_attribute]
                
                if label_cluster in result_dict:
                    
                    #retrieve depth of the corresponding label in that graph
                    depth_value = graph.nodes[node]['depth']
                    
                    result_dict[label_cluster].append((i_graph, node))
                    label_depths[label_cluster].append(depth_value)
                    
                else:
                    #retrieve depth of the corresponding label in that graph
                    depth_value = graph.nodes[node]['depth']
                    
                    result_dict[label_cluster] = [(i_graph, node)]
                    label_depths[label_cluster] = [depth_value]


    return result_dict,label_depths


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    #path_to_graphs = '../data/OASIS_labelled_pits_graphs'
    path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs'

    vmax = 0.7
    #methods = ['media','neuroimage']#'mALS']

    methods = ['mALS','mSync','CAO','kerGM','MatchEig','media','neuroimage']

    #methods = ['neuroimage']

    trash_label = -2#-0.1#-2
    reg_or_unreg = ''#'_unreg'#''
    largest_ind = 22#24
    default_label = -0.1
    nb_bins = 20
    dens = False

    list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)




    simbs = ['cross','ring','disc','square']

    for ind, method in enumerate(methods):
        print('------------'+method+'---------------')
        vb_sc = gv.visbrain_plot(mesh)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        if 'media' in method:
            label_attribute = 'label_media'
        elif 'neuroimage' in method:
            label_attribute = 'label_neuroimage'

        else:
            label_attribute = 'labelling_' + method + reg_or_unreg

        print(label_attribute)
        cluster_dict,depth_dict = create_clusters_lists(list_graphs, label_attribute=label_attribute)

        #calculate depth variance
        depth_dict_var = {}

        for k in depth_dict:
            mean_var = np.std(depth_dict[k])
            depth_dict_var[k] = mean_var

        # Calculate the centroid
        centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")
        centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, mesh, attribute_vertex_index='ico100_7_vertex_index')

        print('Min node data: ',np.min(list(depth_dict_var.values())))
        print('Max node data: ',np.max(list(depth_dict_var.values())))

        # To plot multiple methods
  
        # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(depth_dict_var.values())),
        #                                                 nodes_size=22, nodes_mask=None, c_map='gist_heat', symbol=simbs[ind],
        #                                                 vmin=0, vmax=1)


        # To plot individual methods

        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(depth_dict_var.values())),
                                                        nodes_size=30, nodes_mask=None, c_map='jet',vmin=0, vmax=vmax)

        vb_sc.add_to_subplot(s_obj)

        #vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=300)
        vb_sc.preview()

