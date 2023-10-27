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
    template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_labelled_pits_graphs'

    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    mesh = sio.load_mesh(template_mesh)
    largest_ind=24
    g_l=list_graphs[largest_ind]#p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(largest_ind)+".gpickle","rb"))
    color_label_ordered = gca.label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm=p.load(open("/mnt/data/work/python_sandBox/Graph_matching/data/r_perm.gpickle","rb"))
    color_label = color_label_ordered[r_perm]

    reg_mesh = gv.reg_mesh(mesh)
    vb_sc1 = gv.visbrain_plot(reg_mesh)
    vb_sc2 = gv.visbrain_plot(reg_mesh)
    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=None, nodes_mask=None, c_map='nipy_spectral')
        vb_sc1.add_to_subplot(s_obj)

        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index_noreg', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=None, nodes_mask=None, c_map='nipy_spectral', symbol='+')
        vb_sc2.add_to_subplot(s_obj)
    vb_sc1.preview()
    vb_sc2.preview()

