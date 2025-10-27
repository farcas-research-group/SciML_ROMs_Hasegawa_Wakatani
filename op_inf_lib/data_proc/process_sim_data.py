from op_inf_lib.data_proc._basics import *
from op_inf_lib.data_proc.gems_lifting import *

def read_raw_snapshots(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end, primitive_var_list, n_cells):
	iterator 		= range(snapshot_time_start, snapshot_time_end + 1)
	n_snapshots 	= len(iterator)
	
	# collect snapshots
	for iter_count, iter_num in enumerate(iterator):

		print('reading snapshot ', data_loc(iter_num))
	
		# load data
		data_set  	= tec.data.load_tecplot_szl([mesh_loc, data_loc(iter_num)], read_data_option=2) 	# data_set is overarching data object
		t_zone    	= data_set.zone(0)																	# for this data, all variable data stored in Zone 0
		zone_vars 	= list(t_zone.dataset.variables())
		

		# t_zone.dataset.variable("Density [[]kg/m^3[]]").name = 'Density'
		# t_zone.dataset.variable("Pressure [[]Pa[]]").name = 'Pressure'
		# t_zone.dataset.variable("Temperature [[]K[]]").name = 'Temperature'
		# t_zone.dataset.variable("Subgrid Kinetic Energy [[]m^2/s^2[]]").name = 'Subgrid Kinetic Energy'
		# t_zone.dataset.variable("U-Velocity [[]m/s[]]").name = 'U-Velocity'
		# t_zone.dataset.variable("V-Velocity [[]m/s[]]").name = 'V-Velocity'
		# t_zone.dataset.variable("W-Velocity [[]m/s[]]").name = 'W-Velocity'


		# some preliminary info about the data set
		if (iter_num == snapshot_time_start):
			indic			= get_indices(zone_vars, primitive_var_list)
			snapshot_mat 	= np.zeros((n_cells*(len(primitive_var_list)), n_snapshots), dtype=np.float64)

		# extract specified variables
		snapshot 					= read_snapshot(t_zone, n_cells, zone_vars, primitive_var_list, indic)
		snapshot_mat[:, iter_count] = snapshot

	return snapshot_mat

def get_lifted_scaling_data(lifted_snapshots,n_cells):
        ndof 			= lifted_snapshots.shape[0]
        nvars 			= int(ndof/n_cells)
        scaling_data 	= np.zeros((nvars, 4))
        
        for i in range(nvars):
            temp 				= lifted_snapshots[i*n_cells:(i+1)*n_cells,:]
            scaling_data[i,:] 	= [np.min(temp), np.max(temp), np.mean(temp), np.std(temp)]
            
            if np.min(temp) == np.max(temp) == 0.:
                scaling_data[i, :] = [1., 1., 0., 0.]
                
        return scaling_data

def normalize_data(snapshots,transform,trans_data):
	nvars 	= trans_data.shape[0]
	n_cells = int(snapshots.shape[0]/nvars)
	norm_ss = np.zeros(snapshots.shape)
	
	for i in range(nvars):
		norm_ss[i*n_cells:(i+1)*n_cells]  = transform(snapshots[i*n_cells:(i+1)*n_cells,:],trans_data[i,:])
	
	return norm_ss


# this is left in for backwards-compatibility with case 1 things.
# future cases should call "read_raw_snapshots" above and then "lift_raw_data" from the gems_lifting module
def lift_snapshot_matrix(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end, primitive_var_list):

	raw_snapshots, n_cells = read_raw_snapshots(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end,primitive_var_list)

	lifted_snapshots = lift_raw_data(raw_snapshots,n_cells)

	return lifted_snapshots,n_cells

# lift and normalize snaphots
def lift_and_normalize_snapshot_matrix(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end, primitive_var_list, normalize_lin_trans):

	lifted_snapshots,n_cells = lift_snapshot_matrix(data_loc, mesh_loc, snapshot_time_start, snapshot_time_end, primitive_var_list)

	xi, u, v, w, p, rZ, rC = parse_vars(lifted_snapshots,n_cells,'lifted')

	xi_n 	= normalize_lin_trans(xi)
	p_n 	= normalize_lin_trans(p)
	u_n 	= normalize_lin_trans(u)
	v_n 	= normalize_lin_trans(v)
	w_n 	= normalize_lin_trans(w)
	rC_n 	= normalize_lin_trans(rC)
	rZ_n 	= normalize_lin_trans(rZ)

	lifted_and_normalized_snapshots = np.concatenate((xi_n, u_n, v_n, w_n, p_n, rZ_n, rC_n), axis=0)

	return lifted_and_normalized_snapshots


