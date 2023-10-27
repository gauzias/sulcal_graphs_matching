import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import matplotlib.pyplot as plt
import numpy as np
import slam.plot as splt

if __name__ == "__main__":

    #path_to_graphs = '../data/OASIS_full_batch/modified_graphs'
    #path_to_graphs = '../data/simu_graph/' #simulated graphs
    #path_to_graphs = '../data/Oasis_original_new/'  # non registered graphs
    path_to_graphs = '../data/HCP/modified_graphs_left/'

    file_sphere_mesh = '../data/HCP/Q1-Q6_RelatedValidation210.L.sphere.32k_fs_LR.surf.gii'
    sphere_mesh = sio.load_mesh(file_sphere_mesh)


    list_graphs = gp.load_graphs_in_list(path_to_graphs)


    graphs_nodes_number = list()
    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        graphs_nodes_number.append(len(graph.nodes))
        gp.sphere_nearest_neighbor_interpolation(graph, sphere_mesh)

    print('nodes numbers : ')
    print('mean=', np.mean(graphs_nodes_number))
    print('std=', np.std(graphs_nodes_number))
    print('sum across subjects=', np.sum(graphs_nodes_number))

    # using an average mesh as template
    template_mesh = '../data/HCP/Q1-Q6_RelatedValidation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))


    density_map = gv.nodes_density_map(list_graphs, mesh, nb_iter=3, dt=0.5)

    plt.figure()
    plt.hist(density_map, bins=50)
    plt.show()

    # visb_sc = gv.visbrain_plot(mesh=mesh,
    #                         tex=density_map,
    #                         caption='density map',
    #                         cmap="jet",
    #                         #clim=(0, 0.03),
    #                         cblabel='mean curvature')

    visb_sc = splt.visbrain_plot(mesh=mesh, tex=density_map,
                             caption='Template mesh',
                             cblabel='density',
                             cmap = 'jet')

    #visb_sc.preview()

    # visb_sc2 = gv.visbrain_plot(mesh=sphere_mesh,
    #                         tex=density_map,
    #                         caption='density map',
    #                         cmap="jet")

    visb_sc = splt.visbrain_plot(mesh=sphere_mesh, tex=density_map,
                             caption='Sphere mesh',
                             cblabel='density',cmap ='jet', visb_sc=visb_sc)

    visb_sc.preview()





    # ## visualization of nodes as spheres
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
    #
    # # using a sphere mesh as template
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    # mesh = sio.load_mesh(template_mesh)
    #
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
