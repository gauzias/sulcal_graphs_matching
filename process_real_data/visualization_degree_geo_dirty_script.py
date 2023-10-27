import numpy as np
import os
import pickle as p
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import networkx as nx
import tools.graph_visu as gv
import tools.graph_processing as gp
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly.figure_factory as ff

path_1 = "/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs/"


#generate random color codes for plotting
def generate_random_color_codes(num_colors):
    
    number_of_colors = num_colors
    
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    
    return color

def remove_dummy_nodes(graph):
	G = graph.copy()
	to_remove = []
	for (p, d) in G.nodes(data=True):
		if d['is_dummy'] == True:
			to_remove.append(p)
	G.remove_nodes_from(to_remove)
	return G

degree_list = []
for graph in os.listdir(path_1):
    G = p.load(open(path_1 + graph, "rb" )) #read the graphs
    new_G = remove_dummy_nodes(G)
    new_G.remove_edges_from(nx.selfloop_edges(new_G)) # remove self loops
    degree_list.append(list(dict(nx.degree(new_G)).values()))

color = generate_random_color_codes(len(degree_list))
fig = ff.create_distplot(degree_list, os.listdir(path_1), show_hist=False, colors=color)

# Add title
fig.update_layout(title_text='Degree density real graph')
fig.show()


def degree_of_neighb(G):
    nb_degree_dict = {}
    for node in G:
        nb_degree = []
        for nb in G.neighbors(node):
            nb_degree.append(G.degree(nb))
        nb_degree_dict[node] = nb_degree
    return nb_degree_dict

all_graph_nb_degree = []
for graph in os.listdir(path_1):
    graph = nx.read_gpickle(path_1+graph)
    graph = remove_dummy_nodes(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph)) # remove self loops
    all_graph_nb_degree.append(degree_of_neighb(graph))


avg_nb_degree_all_graph = []
for degree_dict in all_graph_nb_degree:
    avg_degree = [np.mean(lst) for lst in list(degree_dict.values())]
    avg_nb_degree_all_graph.append(avg_degree)


color = generate_random_color_codes(len(avg_nb_degree_all_graph))
fig = ff.create_distplot(avg_nb_degree_all_graph, os.listdir(path_1), show_hist=False,bin_size=.2, colors=color)

# Add title
fig.update_layout(title_text='Average neigbour degree for real graphs')
fig.show()


# neighbours geo distance

def geo_dist_of_neighb(G):
    nb_dist_dict = {}
    for node in G:
        nb_geo_dist = []
        for nb in G.neighbors(node):
            nb_geo_dist.append(G.get_edge_data(node,nb)['geodesic_distance'])
        nb_dist_dict[node] = nb_geo_dist
    return nb_dist_dict

all_graph_nb_distance = []
for graph in os.listdir(path_1):
    graph = nx.read_gpickle(path_1+graph)
    graph = remove_dummy_nodes(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph)) # remove self loops
    all_graph_nb_distance.append(geo_dist_of_neighb(graph))


avg_nb_dist_all_graph = []
for dist_dict in all_graph_nb_distance:
    avg_dist = [np.mean(lst) for lst in list(dist_dict.values())]
    avg_nb_dist_all_graph.append(avg_dist)


color = generate_random_color_codes(len(avg_nb_dist_all_graph))
fig = ff.create_distplot(avg_nb_dist_all_graph, os.listdir(path_1), show_hist=False,bin_size=.2, colors=color)

# Add title
fig.update_layout(title_text='Average neigbour geo distance for real graphs')
fig.show()