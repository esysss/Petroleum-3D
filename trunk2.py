import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import mpl_toolkits.mplot3d.axes3d as axes3d
import pandas
import numpy
import networkx
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import pickle

def read(what, have_it):
    if have_it:
        # tensor = np.load('Store/phi.npy')
        tensor = np.load('Store/perm.npy')
    elif what == "excel":
        df = pandas.read_excel('Data/Test.xlsx')
        df = df[['X', 'Y', 'Z', "Phi"]]

        tensor = np.zeros([df['Z'].max(), df['X'].max(), df['Y'].max()])
        tensor.fill(None)

        for i in range(len(df)):
            tensor[int(df.iloc[i]['X']) - 1, int(df.iloc[i]['Y']) - 1, int(df.iloc[i]['Z']) - 1] = df.iloc[i]['Phi']

        print("this tensor is inshape {} !!!".format(tensor.shape))

    elif what == "txt":
        # df = pandas.read_csv('data/Phi.txt',delimiter = "\t", header = None)
        df = pandas.read_csv('data/Perm.txt',delimiter = "\t", header = None)

        df.columns = ['X', 'Y', 'Z', "Phi"]

        tensor = np.zeros([df['Z'].max(), df['X'].max(), df['Y'].max()])
        tensor.fill(None)

        print("Reading the data")
        for i in tqdm(range(len(df))):
            tensor[int(df.iloc[i]['Z']) - 1, int(df.iloc[i]['X']) - 1, int(df.iloc[i]['Y']) - 1] = df.iloc[i]['Phi']

        print("this tensor is inshape {} !!!".format(tensor.shape))

    # np.save("perm.npy",tensor)
    return tensor

def lambdaa(tensor):
    min_of_lambda = tensor.min()
    max_of_lambda = tensor.max()

    # lambda_step = tensor.reshape(1, tensor.shape[0] * tensor.shape[1] * tensor.shape[2])
    # lambda_step = np.abs(np.insert(lambda_step, 0, 0) - np.insert(lambda_step, -1, 0))
    # lambda_step = np.min(lambda_step[np.nonzero(lambda_step)])

    lambda_step = (max_of_lambda - min_of_lambda)/500

    return min_of_lambda, max_of_lambda, lambda_step


def lambda_calculator(array, lam):
    idx1 = 0
    idx2 = idx1 + 1
    start_point = 0
    while True:
        if abs(array[idx1] - array[idx2]) < lam:
            idx1 = idx2
            idx2 += 1
        else:
            array[start_point:idx2] = np.mean(array[start_point:idx2])
            idx1 = idx2
            idx2 = idx1 + 1
            start_point = idx1

        if idx2 >= len(array):
            array[start_point:] = np.mean(array[start_point:])
            break

    return len(np.unique(array))

def lambda_address(tensor, l, X_limit, Y_limit, Z_limit, have_it):

    if have_it:
        # pickleIN = open("Store/dick.pickle","rb") #for the 10,10,10 test (Test.xlsx)
        pickleIN = open("Store/phi.pickle","rb") #for the phi
        # pickleIN = open("Store/perm.pickle","rb") #for the perm
        lambda_addresses = pickle.load(pickleIN)
        pickleIN.close()
        return lambda_addresses

    m = l[0]  # min of lambda
    M = l[1]  # max of lambda
    ls = l[2]  # lambda step

    original_tensor = tensor.copy()
    lambda_addresses = {}  # keys: "row/column/depth","X/Y","Y/Z"  values: (lambda, number of blocks)
    """
    1st = Z axis (depths)
    2nd = X axis (rows)
    3rd = Y axis (columns)
    """
    print("The lambda calculation is in process ...")
    for lambs in tqdm(list(np.arange(m, M, ls))):
        tensor = original_tensor

        # for the Z axis
        for x in range(tensor.shape[1]):
            for y in range(tensor.shape[2]):
                temp = lambda_calculator(original_tensor[:, x, y], lambs)
                if temp > Z_limit:
                    try:
                        if lambda_addresses["depth", x, y][1] > temp:
                            lambda_addresses["depth", x, y] = (lambs, temp)
                    except:
                        lambda_addresses["depth", x, y] = (lambs, temp)

        # for the X axis
        for z in range(tensor.shape[0]):
            for y in range(tensor.shape[2]):
                temp = lambda_calculator(original_tensor[z, :, y], lambs)
                if temp > X_limit:
                    try:
                        if lambda_addresses["row", z, y][1] > temp:
                            lambda_addresses["row", z, y] = (lambs, temp)
                    except:
                        lambda_addresses["row", z, y] = (lambs, temp)

        # for the Y axis
        for z in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                temp = lambda_calculator(original_tensor[z, x, :], lambs)
                if temp > Y_limit:
                    try:
                        if lambda_addresses["column", z, x][1] > temp:
                            lambda_addresses["column", z, x] = (lambs, temp)
                    except:
                        lambda_addresses["column", z, x] = (lambs, temp)


    theFile = open("Store/perm.pickle", "wb")  # it says to write in bite
    pickle.dump(lambda_addresses, theFile)
    theFile.close()

    return lambda_addresses

def distance_calculator(tensor):
    """
    depth_tensor is the distance between two neighbors in different depths like depth 1 and depth 2 in the point (0,0)
    row_tensor in the distance between two neighbors in  different rows like above row and blow row (up to down)
    """
    depth_tensor = np.abs(tensor - np.concatenate((tensor[1:,:,:], tensor[0,:,:].reshape(1, tensor.shape[1], tensor.shape[2])), axis=0))[:-1,:,:]
    row_tensor = np.abs(tensor - np.concatenate((tensor[:,1:,:], tensor[:,0,:].reshape(tensor.shape[0], 1, tensor.shape[2])), axis=1))[:,:-1,:]
    column_tensor = np.abs(tensor - np.concatenate((tensor[:,:,1:], tensor[:,:,0].reshape(tensor.shape[0], tensor.shape[1], 1)),axis=2))[:,:,:-1]

    return (depth_tensor, row_tensor, column_tensor)

def best_next(tensor, lambda_addresses, forbid_list):

    distances = distance_calculator(tensor)

    axis = [0,0,0]

    depth_min = np.sort(distances[0][np.nonzero(distances[0])])
    row_min = np.sort(distances[1][np.nonzero(distances[1])])
    column_min = np.sort(distances[2][np.nonzero(distances[2])])

    while True:
        try:
            d_min = depth_min[axis[0]]
        except:
            d_min = np.inf

        try:
            r_min = row_min[axis[1]]
        except:
            r_min = np.inf

        try:
            c_min = column_min[axis[2]]
        except:
            c_min = np.inf

        if d_min == np.inf and r_min == np.inf and c_min == np.inf:
            break

        min_of_mins, lambdas, addresses = next_item(lambda_addresses, distances, d_min, r_min, c_min)

        for add,lam in zip(addresses,lambdas):
            if not [int(min_of_mins),add] in forbid_list:
                return min_of_mins,lam, add

        axis[min_of_mins] += 1

    print("all done")
    return "finished", add, lam

def next_item(lambda_addresses, distances, depth_min, row_min, column_min):

    min_of_mins = np.argmin([depth_min, row_min, column_min])
    addresses = []
    lambdas = []
    if min_of_mins == 0:  # is it in the direction of depths?
        temp_address = np.matrix(np.where(distances[0] == depth_min))

        for i in range(temp_address.shape[1]):
            temp = temp_address[:,i]
            addresses.append((int(temp[0]), int(temp[1]), int(temp[2])))

        for i in addresses:
            lambdas.append(lambda_addresses["depth", i[1], i[2]])

    elif min_of_mins == 1:  # is it in the direction of rows?
        temp_address = np.matrix(np.where(distances[1] == row_min))

        for i in range(temp_address.shape[1]):
            temp = temp_address[:,i]
            addresses.append((int(temp[0]), int(temp[1]), int(temp[2])))

        for i in addresses:
            lambdas.append(lambda_addresses["row", i[0], i[2]])

    else:  # is it in the direction of columns?
        temp_address = np.matrix(np.where(distances[2] == column_min))

        for i in range(temp_address.shape[1]):
            temp = temp_address[:,i]
            addresses.append((int(temp[0]), int(temp[1]), int(temp[2])))

        for i in addresses:
            lambdas.append(lambda_addresses["column", i[0], i[1]])

    return min_of_mins, lambdas, addresses

def merge(axis, tensor, flag_tensor, lambdaa, address, counter, limit):

    forbidden = False

    neighbor_address = list(address)
    if axis == 0:
        neighbor_address[0] += 1

    elif axis == 1:
        neighbor_address[1] += 1

    else:
        neighbor_address[2] += 1
    neighbor_address = tuple(neighbor_address)

    entry_flag = flag_tensor[address]
    neighbor_flag = flag_tensor[neighbor_address]
    to_check = abs(tensor[address] - tensor[neighbor_address])

    block1 = len(flag_tensor[flag_tensor == entry_flag])
    block2 = len(flag_tensor[flag_tensor == neighbor_flag])

    if (entry_flag != 0 and block1+1 > limit) or (neighbor_flag !=0 and block2+1 > limit) or (entry_flag !=0 and neighbor_flag !=0 and block1+block2 > limit):
        forbidden = True

    elif to_check <= lambdaa:
        print("merged")
        counter += 1
        if entry_flag == 0:
            flag_tensor[address] = counter
        else:
            flag_tensor[np.where(flag_tensor == entry_flag)] = counter

        if neighbor_flag == 0:
            flag_tensor[neighbor_address] = counter
        else:
            flag_tensor[np.where(flag_tensor == neighbor_flag)] = counter

        tensor[np.where(flag_tensor == counter)] = np.mean(tensor[np.where(flag_tensor == counter)])

    else:
        forbidden = True

    return tensor, counter, flag_tensor, forbidden

def plot(flag):
    def cube_marginals(cube, normalize=False):
        c_fcn = np.mean if normalize else np.sum
        xy = c_fcn(cube, axis=0)
        xz = c_fcn(cube, axis=1)
        yz = c_fcn(cube, axis=2)
        return (xy, xz, yz)

    # colors = ['co', 'ro', 'bo', 'go', 'mo', 'yo', 'ko', 'wo']


    cube = np.zeros(flag.shape)

    x = None
    y = None
    z = None,
    normalize = False
    plot_front = True

    """Use contourf to plot cube marginals"""
    (Z, Y, X) = cube.shape
    (xy, xz, yz) = cube_marginals(cube, normalize=normalize)
    if x == None: x = np.arange(X)
    if y == None: y = np.arange(Y)
    if z == None: z = np.arange(Z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plt.title("the merged tensor")
    ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [0, 0, 0, 0, 0], 'k-')
    ax.plot([0, X - 1, X - 1, 0, 0], [0, 0, Y - 1, Y - 1, 0], [Z - 1, Z - 1, Z - 1, Z - 1, Z - 1], 'k-')
    ax.plot([0, 0], [0, 0], [0, Z - 1], 'k-')
    ax.plot([X - 1, X - 1], [0, 0], [0, Z - 1], 'k-')
    ax.plot([X - 1, X - 1], [Y - 1, Y - 1], [0, Z - 1], 'k-')
    ax.plot([0, 0], [Y - 1, Y - 1], [0, Z - 1], 'k-')

    colors = tuple(mcd.CSS4_COLORS.values())

    for j,i in enumerate(np.unique(flag)):
        comm = np.matrix(np.where(flag == i))
        for cc in range(comm.shape[1]):
            c = comm[:,cc]
            ax.plot([int(c[0]),int(c[0])],[int(c[1]),int(c[1])],[int(c[2]),int(c[2])], color = colors[j],marker = 'o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()