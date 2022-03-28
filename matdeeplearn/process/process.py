import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate

##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree

################################################################################
# Data splitting
################################################################################

##basic train, val, test split
def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")


##Basic CV split
def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1e6), save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length :", fold_length, "unused length:", unused_length, "seed", seed)
    return cv_dataset[0:num_folds]


################################################################################
# Pytorch datasets
################################################################################

##Fetch dataset; processes the raw data if specified
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    transforms = GetY(index=target_index)

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )        
    return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


##Dataset class from pytorch/pytorch geometric
class StructureDataset_large(Dataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data


################################################################################
#  Processing
################################################################################


def process_data(data_path, processed_path, processing_args):

    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    ##Load dictionary
    if processing_args["dictionary_source"] != "generated":
        if processing_args["dictionary_source"] == "default":
            print("Using default dictionary.")
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "dictionary_default.json",
                )
            )
        elif processing_args["dictionary_source"] == "blank":
            print(
                "Using blank dictionary. Warning: only do this if you know what you are doing"
            )
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "dictionary_blank.json"
                )
            )
        else:
            dictionary_file_path = os.path.join(
                data_path, processing_args["dictionary_path"]
            )
            if os.path.exists(dictionary_file_path) == False:
                print("Atom dictionary not found, exiting program...")
                sys.exit()
            else:
                print("Loading atom dictionary from file.")
                atom_dictionary = get_dictionary(dictionary_file_path)

    ##Load targets
    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), (
        "targets not found in " + target_property_file
    )
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    ##Read db file if specified
    ase_crystal_list = []
    if processing_args["data_format"] == "db":
        db = ase.db.connect(os.path.join(data_path, "data.db"))
        row_count = 0
        # target_data=[]
        for row in db.select():
            # target_data.append([str(row_count), row.get('target')])
            ase_temp = row.toatoms()
            ase_crystal_list.append(ase_temp)
            row_count = row_count + 1
            if row_count % 500 == 0:
                print("db processed: ", row_count)

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(target_data)):

        structure_id = target_data[index][0]
        data = Data()

        ##Read in structure file using ase
        if processing_args["data_format"] != "db":
            ase_crystal = ase.io.read(
                os.path.join(
                    data_path, structure_id + "." + processing_args["data_format"]
                )
            )
            data.ase = ase_crystal
        else:
            ase_crystal = ase_crystal_list[index]
            data.ase = ase_crystal

        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        ##Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (
                distance_matrix_trimmed.fill_diagonal_(1) != 0
            ).int()
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask
        
        #Smooth and interpolate data to specified length
        import scipy 
        from scipy import interpolate
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter1d
        
        dos_read = np.load(os.path.join(data_path, structure_id + ".npy"))         
        dos_sum = np.sum(dos_read[:,1:,:], axis=1)   
        np.savetxt((os.path.join(data_path, structure_id + "_processed_0.csv")), np.concatenate((dos_read[0,0,:][np.newaxis,:], dos_sum), axis=0), delimiter=",") 
          
        for i in range(0, len(ase_crystal)):
            dos_sum[i,:] = gaussian_filter1d(dos_sum[i,:], sigma=7)         
        
        #interpolation and shift energy window             
        dos_length=400
        dos=np.zeros((len(ase_crystal),dos_length))
        for i in range(0, len(ase_crystal)):
            xfit=dos_read[i,0,:]
            yfit=dos_sum[i,:]                     
            dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
            xnew = np.linspace(-10, 10, dos_length) 
            dos[i,:]=dos_fit(xnew)  
        data.y = torch.Tensor(dos)
        np.savetxt((os.path.join(data_path, structure_id + "_processed_1.csv")), dos, delimiter=",") 
        
        dos_features = get_dos_features(torch.Tensor(xnew), data.y)
        data.dos_features = torch.Tensor(dos_features)
        
        scaling = np.max(dos, axis=1)      
        for i in range(0, len(ase_crystal)):
            dos[i,:] = dos[i,:]/scaling[i] 

        #if 0 in scaling:
        #    print(structure_id)
        #if np.isnan(np.min(dos)):
        #    print(structure_id)

        np.savetxt((os.path.join(data_path, structure_id + "_processed_final.csv")), dos, delimiter=",")        
        data.dos_scaled=torch.Tensor(dos)       
        data.scaling_factor = torch.Tensor(scaling)
        
        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data.structure_id = [[structure_id] * len(ase_crystal)]

        if processing_args["verbose"] == "True" and (
            (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))
            # if index == 0:
            # print(data)
            # print(data.edge_weight, data.edge_attr[0])

        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    ##Generate node features
    if processing_args["dictionary_source"] != "generated":
        ##Atom features(node features) from atom dictionary file
        for index in range(0, len(data_list)):
            atom_fea = np.vstack(
                [
                    atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                    for i in range(len(data_list[index].ase))
                ]
            ).astype(float)
            data_list[index].x = torch.Tensor(atom_fea)
    elif processing_args["dictionary_source"] == "generated":
        ##Generates one-hot node features rather than using dict file
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        lb.fit(species)
        for index in range(0, len(data_list)):
            data_list[index].x = torch.Tensor(
                lb.transform(data_list[index].ase.get_chemical_symbols())
            )

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    ##makes SOAP and SM features from dscribe
    if processing_args["SOAP_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SOAP
            
        make_feature_SOAP = SOAP(
            species=species,
            rcut=processing_args["SOAP_rcut"],
            nmax=processing_args["SOAP_nmax"],
            lmax=processing_args["SOAP_lmax"],
            sigma=processing_args["SOAP_sigma"],
            periodic=periodicity,
            sparse=False,
            rbf="gto",
            crossover=True,
        )
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = torch.Tensor(features_SOAP)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SOAP length: ",
                        features_SOAP.shape,
                    )
                print("SOAP descriptor processed: ", index)

    if processing_args["LMBTR_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import LMBTR
        
        make_feature_LMBTR = LMBTR(
            species=species,
            k2 = {"geometry": {"function": "distance"},"grid": {"min": 0, "max": processing_args["LMBTR_rcut"], "sigma" : processing_args["LMBTR_sigma"], "n" : processing_args["LMBTR_grid"]},"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}},
            k3 = {"geometry": {"function": "angle"},"grid": {"min": 0, "max": 180, "sigma" : processing_args["LMBTR_sigma"], "n": processing_args["LMBTR_grid"]},"weighting" : {"function": "exp", "scale": 0.5, "threshold": 1e-3}},
            periodic=True,
            sparse=False,
            flatten=True,
        )
            
        for index in range(0, len(data_list)):
            features_LMBTR = make_feature_LMBTR.create(data_list[index].ase)
            data_list[index].extra_features_LMBTR = torch.Tensor(features_LMBTR)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "LMBTR length: ",
                        features_LMBTR.shape,
                    )
                print("LMBTR descriptor processed: ", index)

    ##Generate edge features
    if processing_args["edge_features"] == "True":

        ##Distance descriptor using a Gaussian basis
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
        # print(GetRanges(data_list, 'distance'))
        NormalizeEdge(data_list, "distance")
        # print(GetRanges(data_list, 'distance'))
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )
            if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
            ):
                print("Edge processed: ", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    ##Save processed dataset to file
    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))

    elif processing_args["dataset_type"] == "large":
        for i in range(0, len(data_list)):
            torch.save(
                data_list[i],
                os.path.join(
                    os.path.join(data_path, processed_path), "data_{}.pt".format(i)
                ),
            )


################################################################################
#  Processing sub-functions
################################################################################

##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


##Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


##get dos features
def get_dos_features(x, dos):
    dos=torch.abs(dos) 

    center=torch.sum(x*dos, axis=1)/torch.sum(dos, axis=1)
    x_offset = torch.repeat_interleave(x[np.newaxis,:], dos.shape[0], axis=0)-center[:,None]
    width = torch.diagonal(torch.mm((x_offset**2), dos.T))/torch.sum(dos, axis=1)
    skew = torch.diagonal(torch.mm((x_offset**3), dos.T))/torch.sum(dos, axis=1)/width**(1.5)
    kurtosis = torch.diagonal(torch.mm((x_offset**4), dos.T))/torch.sum(dos, axis=1)/width**(2)
    
    #find zero index (fermi leve)
    zero_index = torch.abs(x-0).argmin().long()
    ef_states = torch.sum(dos[:,zero_index-20:zero_index+20], axis=1)*abs(x[0]-x[1])
    return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)


##Obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)


################################################################################
#  Transforms
################################################################################

##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data
