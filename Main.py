import trunk as t
import numpy as np
import pickle
from tqdm import tqdm


#read the data (3d)
tensor = t.read('txt',have_it = True)

#get the lambda min/max and lambda_step
l = t.lambdaa(tensor)

#limit of blocks (how much they can merge in those directions)
X_limit = np.floor(.3 * tensor.shape[1])
Y_limit = np.floor(.3 * tensor.shape[2])
Z_limit = np.floor(.3 * tensor.shape[0])

N_limit = 100 # more than 100 blocks can not be merged

lambda_addresses = t.lambda_address(tensor, l, X_limit, Y_limit, Z_limit, have_it = True)

flag_tensor = np.zeros(tensor.shape)
block_counter = 0   #to use it in the flag_tensor

forbid_list = []

counter_saver = 0

total_cells = len(tensor.flatten())
for i in range(total_cells):
    axis, temp_lambda, temp_address = t.best_next(tensor, lambda_addresses, forbid_list)

    if axis == "finished":
        break
    tensor, block_counter,flag_tensor, forbidden = t.merge(
        axis, tensor, flag_tensor, temp_lambda[0], temp_address, block_counter, N_limit)

    if forbidden:
        forbid_list.append([axis,temp_address])

    counter_saver+=1
    if counter_saver%100 == 0:
        results = {"tensor":tensor, "flag":flag_tensor}

        # theFile = open("Store/perm_results.p", "wb")
        theFile = open("phi_results.p", "wb")
        pickle.dump(results, theFile)
        theFile.close()

    print(len(np.unique(flag_tensor))-1)

results = {"tensor": tensor, "flag": flag_tensor}

# theFile = open("Store/perm_results.p", "wb")
theFile = open("phi_results.p", "wb")
pickle.dump(results, theFile)
theFile.close()

print("finished")