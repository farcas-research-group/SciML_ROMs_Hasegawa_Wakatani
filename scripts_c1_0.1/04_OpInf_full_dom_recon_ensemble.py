from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

import xarray as xr

from config.HW import *

r = 78

ENGINE 	= "h5netcdf"

fh 	    = xr.open_dataset('/storage1/HW/paper/0.10_300_snapshots.h5', engine=ENGINE)
Q_test = (xr.concat([fh["density"].expand_dims(dim={"field": ["density"]}, axis=1), \
	       fh["potential"].expand_dims(dim={"field": ["potential"]}, axis=1), ], \
	       dim="field",).stack(n=("field", "y", "x")).transpose("n", "time").data)


no_time_steps_rec = Q_test.shape[1]

time_steps_rec = [0 + 500*i for i in range(no_time_steps_rec)]

data = np.load('/storage1/HW/paper/res_c1_0.1/postprocessing_ensemble_c1_5.0_training_end' + str(training_size) + '_r'+str(r)+'.npz')

Xrec = data['X_OpInf']

print(Xrec.shape)


temp 	= np.load(POD_file)
Ur  	= temp['Vr'][:, :r]
S       = temp['S']
temp    = 0

# np.save('results/svals_RDE_360_case_1.npy', S)



# reconstruct, parse
X    = Ur @ Xrec.T[:, time_steps_rec]

np.save('results/rec_full_fields_c1_0.1_r' + str(r) + '.npy', X)
np.save('results/ref_full_fields_c1_0.1_r' + str(r) + '.npy', Q_test)
