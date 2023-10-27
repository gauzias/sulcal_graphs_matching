import sys
import os
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p



def farthest_point_sampling(coords):
    dist_mat =np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    N = coords.shape[0]

    maxdist  = 0
    bestpair = ()
    for i in range(N):
      for j in range(i+1,N):
        if dist_mat[i,j]>maxdist:
          maxdist = dist_mat[i,j]
          bestpair = (i,j)

    P = list()
    P.append(bestpair[0])
    P.append(bestpair[1])

    while len(P)<N:
        maxdist = 0
        vbest = None
        for v in range(N):
            if v in P:
                continue
            for vprime in P:
                if dist_mat[v,vprime]>maxdist:
                    maxdist = dist_mat[v,vprime]
                    vbest   = v
        P.append(vbest)

    return P


if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
#'../data/template_mesh/OASIS_avg.lh.white.talairach.unreg.ico7.gii'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_match_mat = '../data/Oasis_original_new_with_dummy/'

    cmap = gv.rand_cmap(101, type='bright', first_color_black=True, last_color_black=False, verbose=True)#'nipy_spectral'#'gist_ncar'#'nipy_spectral'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    cmap = 'gist_ncar'

    ico_mesh = sio.load_mesh('../data/template_mesh/ico100_7.gii')
    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)

    largest_ind=22
    g = list_graphs[largest_ind]
    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', ico_mesh)
    color_label_ordered = gca.label_nodes_according_to_coord(g, reg_mesh, coord_dim=0)
    color_label_ordered = color_label_ordered-0.1
    #r_perm = p.load(open("../data/r_perm.gpickle","rb"))
    r_perm = np.random.permutation(len(g))
    p.dump(r_perm, open("../data/r_perm_22.gpickle", "wb"))
    color_label_r = color_label_ordered[r_perm]

    farthest_reordering = farthest_point_sampling(nodes_coords)
    color_label = color_label_ordered[farthest_reordering]

    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
    #color_label = np.array([l for l in labels])
    vb_sc = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_ordered, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap,  vmin=-0.1, vmax=1)
    vb_sc.add_to_subplot(s_obj)


    vb_sc1 = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap,  vmin=-0.1, vmax=1)
    vb_sc1.add_to_subplot(s_obj)


    vb_sc2 = gv.visbrain_plot(reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_r, nodes_size=60,
                                                    nodes_mask=None, c_map=cmap,  vmin=-0.1, vmax=1)
    vb_sc2.add_to_subplot(s_obj)

    vb_sc.preview()
    vb_sc1.preview()
    vb_sc2.preview()