from op_inf_lib.data_proc._basics import *

# get snapshot matrix
def get_raw_simulation_snapshots(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end, primitive_var_list):

	iterator 		= range(snapshot_time_start, snapshot_time_end + 1)
	n_snapshots 	= len(iterator)
	
	# collect snapshots
	for iter_count, iter_num in enumerate(iterator):
	
		# load data
		data_set  	= tec.data.load_tecplot_szl([mesh_loc, data_loc(iter_num)], read_data_option=2) 
		# data_set  	= tec.data.load_tecplot_szl(data_loc(iter_num), read_data_option=2) 

		t_zone    	= data_set.zone(0)																	
		zone_vars 	= list(t_zone.dataset.variables())

		# some preliminary info about the data set
		if (iter_num == snapshot_time_start):
			n_cells  		= t_zone.num_elements	
			indic			= get_indices(zone_vars, primitive_var_list)
			snapshot_mat 	= np.zeros((n_cells*len(primitive_var_list), n_snapshots), dtype=np.float64)

		# extract specified variables
		snapshot 					= read_snapshot(t_zone, n_cells, zone_vars, primitive_var_list, indic)
		snapshot_mat[:, iter_count] = snapshot

	return snapshot_mat

# read only one snapshot (non-flattened!)
def read_one_snapshot(ex_data_file, mesh_loc, var_list):

	data_set  	= tec.data.load_tecplot_szl([mesh_loc, ex_data_file], read_data_option=2) 	# data_set is overarching data object
	t_zone    	= data_set.zone(0)																	# for this data, all variable data stored in Zone 0
	zone_vars 	= list(t_zone.dataset.variables())

#	t_zone.dataset.variable("Density [[]kg/m^3[]]").name = 'Density'
#	t_zone.dataset.variable("Pressure [[]Pa[]]").name = 'Pressure'
#	t_zone.dataset.variable("Temperature [[]K[]]").name = 'Temperature'
#	t_zone.dataset.variable("Subgrid Kinetic Energy [[]m^2/s^2[]]").name = 'Subgrid Kinetic Energy'
#	t_zone.dataset.variable("U-Velocity [[]m/s[]]").name = 'U-Velocity'
#	t_zone.dataset.variable("V-Velocity [[]m/s[]]").name = 'V-Velocity'
#	t_zone.dataset.variable("W-Velocity [[]m/s[]]").name = 'W-Velocity'

	# some preliminary info about the data set
	n_cells  		= t_zone.num_elements	
	indic			= get_indices(zone_vars, var_list); n_cells = 4204200
	
	# extract specified variables
	snapshot = np.zeros((n_cells, len(var_list)), dtype=np.float64)
	
	for j in indic:
		var 				= zone_vars[j]
		idx 				= var_list.index(var.name)

		snapshot[:, idx] 	= t_zone.values(var.name).as_numpy_array().astype(np.float64)

	return snapshot


# read only one snapshot (non-flattened!)
def read_one_snapshot_ROM(data_file, mesh_loc, var_list):

	data_set  	= tec.data.load_tecplot_szl(data_file, read_data_option=2) 	# data_set is overarching data object
	t_zone    	= data_set.zone(0)																	# for this data, all variable data stored in Zone 0
	zone_vars 	= list(t_zone.dataset.variables())

	# some preliminary info about the data set
	n_cells  		= t_zone.num_elements	
	indic			= get_indices(zone_vars, var_list)
	
	# extract specified variables
	snapshot = np.zeros((n_cells, len(var_list)), dtype=np.float64)
	
	for j in indic:
		var 				= zone_vars[j]
		idx 				= var_list.index(var.name)
		snapshot[:, idx] 	= t_zone.values(var.name).as_numpy_array().astype(np.float64)

	return snapshot
	
	
