from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *

from config.HW import *

if __name__ == "__main__":
    temp = np.load(POD_file)
    S = temp["S"]
    temp = 0

    np.save("results/POD_svals.npy", S)
