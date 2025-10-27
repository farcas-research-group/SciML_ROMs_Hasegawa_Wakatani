import numpy as np
import tecplot as tec
import os

# extract desired variables from snapshot
def read_snapshot(t_zone, n_cells, zone_vars, primitive_var_list, indic):
	
	snapshot = np.zeros((n_cells, len(primitive_var_list)), dtype=np.float64)
	
	for j in indic:
		var 				= zone_vars[j]
		idx 				= primitive_var_list.index(var.name)
		snapshot[:, idx] 	= t_zone.values(var.name).as_numpy_array().astype(np.float64)

	snapshot = snapshot.flatten(order='F')
	
	return snapshot

# get variable indices in Tecplot zone 	
def get_indices(zone_vars, primitive_var_list):

	indices = []
	for j, var in enumerate(zone_vars):
		if var.name in primitive_var_list:
			indices.append(j)
	
	return indices