import os
import sys
#sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
#from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
#from graph_generation.load_graphs_and_create_metadata import dataset_metadata
#from graph_matching_tools.metrics import matching
#import matplotlib.pyplot as plt
import slam.io as sio
#import scipy.io as sco
import tools.graph_processing as gp
import scipy.stats as stats
import tools.graph_visu as gv
import tools.clusters_analysis as gca
import math


def create_clusters_lists_with_label_gender(list_graphs,gender_corresp,label_attribute="label_dbscan"):

	result_dict = {}
	label_depths = {}
	label_gender = {}

	for i_graph, graph in enumerate(list_graphs):
		for node in graph.nodes:
			if not graph.nodes[node]["is_dummy"]:
				label_cluster = graph.nodes[node][label_attribute]
				
				if label_cluster in result_dict:
					
					#retrieve depth of the corresponding label in that graph
					depth_value = graph.nodes[node]['depth']
					
					result_dict[label_cluster].append((i_graph, node))
					label_depths[label_cluster].append(depth_value)
					label_gender[label_cluster].append(gender_corresp[i_graph])
					
				else:
					#retrieve depth of the corresponding label in that graph
					depth_value = graph.nodes[node]['depth']
					
					result_dict[label_cluster] = [(i_graph, node)]
					label_depths[label_cluster] = [depth_value]
					label_gender[label_cluster] = [gender_corresp[i_graph]]


	return result_dict,label_depths,label_gender


def seperate_groups_by_label(label_gender,label_depths):
	
	# Separate groups by label
	label_gen_sep = []
	for key in label_gender.keys():
		M = []
		F = []
		for i in range(len(label_gender[key])):

			if label_gender[key][i] == 'F':

				F.append(label_depths[key][i])
			else:
				M.append(label_depths[key][i])
				
		# 1st list M, 2nd F
		label_gen_sep.append([M,F])
		
	return label_gen_sep


def calculate_tstats_and_pvalues(corresp ,method = 'labelling_mALS'):
	
	# get labeled groups and depths    
	result_dict,label_depths,label_gender = create_clusters_lists_with_label_gender(labeled_graphs,corresp,method)
	
	# depth seperated by groups
	label_gen_sep = seperate_groups_by_label(label_gender,label_depths)
	
	t_stats = {}

	for key,lst in zip(label_gender.keys(),label_gen_sep):

		res = stats.ttest_ind(a=lst[0], b=lst[1], equal_var=True)

		t_stats[key] = [res[0],res[1]]
		
	return t_stats


def drop_nan_get_tstats(t_stat_dict, centroid_dict):

	# drop t,p-values with nan for labels in both centroid and t stst dict.
	
	drop_k = []
	for k in t_stat_dict.keys():
		if math.isnan(t_stat_dict[k][0]):
			drop_k.append(k)

	for k in drop_k:
		t_stat_dict.pop(k)
		centroid_dict.pop(k)
	
	return  np.array(list(t_stat_dict.values())), centroid_dict


if __name__ == '__main__':


	path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs'
	path_ro_correspondence = '../Matlab_MGM_affintiy_gen/gender_correspondence.pickle'

	labeled_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)
	gender_corresp = np.array(nx.read_gpickle(path_ro_correspondence))[:,2] # gender correp list

	# tstats_mALS = calculate_tstats_and_pvalues(gender_corresp, 'labelling_mALS')
	# tstats_mSync = calculate_tstats_and_pvalues(gender_corresp, 'labelling_mSync')
	# tstats_matcheig = calculate_tstats_and_pvalues(gender_corresp, 'labelling_MatchEig')
	# tstats_CAO = calculate_tstats_and_pvalues(gender_corresp, 'labelling_CAO')
	# tstats_kerGM = calculate_tstats_and_pvalues(gender_corresp, 'labelling_kerGM')


	template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
	mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

	vb_sc = gv.visbrain_plot(mesh)
	visb_sc_shape = gv.get_visb_sc_shape(vb_sc)

	simbs = ['cross','ring','disc','square']

	#methods = ['mSync','CAO','kerGM','MatchEig','media','neuroimage']
	#'mALS',
	methods = ['neuroimage']

	for ind, method in enumerate(methods):

		if 'media' in method:
			label_attribute = 'label_media'
		elif 'neuroimage' in method:
			label_attribute = 'label_neuroimage'

		else:
			label_attribute = 'labelling_' + method

		print(label_attribute)
		cluster_dict, label_depths, label_gender = create_clusters_lists_with_label_gender(labeled_graphs,gender_corresp, label_attribute=label_attribute)
		centroid_dict = gca.get_centroid_clusters(labeled_graphs, cluster_dict, coords_attribute="sphere_3dcoords")
		centroids_3Dpos = gca.get_centroids_coords(centroid_dict, labeled_graphs, mesh, attribute_vertex_index='ico100_7_vertex_index')

		## Change tstat according to method
		tstats = calculate_tstats_and_pvalues(gender_corresp, label_attribute)
		t_stat, centroid_dict = drop_nan_get_tstats(tstats, centroid_dict)
		data_plot = t_stat[:, 0]

		vmin = np.min(data_plot)
		vmax = np.max(data_plot)
		print('Min node data: ', vmin)
		print('Max node data: ', vmax)
		print('avg node data: ', np.mean(data_plot))
		print('std node data: ', np.std(data_plot))

		print(len(centroids_3Dpos))
		print(len(data_plot))

		s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data= data_plot,
														nodes_size=30, nodes_mask=None, c_map='gist_heat',vmin=vmin, vmax=vmax)
		vb_sc.add_to_subplot(s_obj)

	vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=300)
	vb_sc.preview()









