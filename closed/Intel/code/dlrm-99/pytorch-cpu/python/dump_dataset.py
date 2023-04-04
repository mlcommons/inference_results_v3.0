import sys
import os
import math
import numpy as np

data_path = sys.argv[1]
output_path = sys.argv[2]
np.random.seed(123)


def calc_criteo_size(data_file, batch_size=1, max_ind_range=-1, bytes_per_feature=4):
    tar_fea = 1
    den_fea = 13
    spa_fea = 26
    tad_fea = tar_fea + den_fea
    tot_fea = tad_fea + spa_fea

    bytes_per_entry = (bytes_per_feature * tot_fea * batch_size)

    num_entries = math.ceil(os.path.getsize(data_file) / bytes_per_entry)

    return num_entries


def generate_sample_partition(
        data_path,
        dump_path,
        samples_to_aggregate_quantile_file,
        max_ind_range=-1):
    random_offsets = []
    samples_to_aggregate = 1
    samples_to_aggregate_quantile_file = sys.path[0] + "/" + samples_to_aggregate_quantile_file

    test_file = data_path + "/terabyte_processed_test.bin"
    num_individual_samples = calc_criteo_size(
        data_file=test_file,
        batch_size = samples_to_aggregate,
        max_ind_range=max_ind_range)

    print("Using variable query size: custom distribution (file " + str(samples_to_aggregate_quantile_file) + ")")
    with open(samples_to_aggregate_quantile_file, 'r') as f:
        line = f.readline()
        quantile = np.fromstring(line, dtype=int, sep=", ")

    l = len(quantile)
    done = False
    qo = 0
    while done == False:
        random_offsets.append(int(qo))
        pr = np.random.randint(low=0, high=l)
        qs = quantile[pr]
        qo = min(qo + qs, num_individual_samples)
        if qo >= num_individual_samples:
            done = True
    random_offsets.append(int(qo))

    # compute min and max number of samples
    nas_max = (num_individual_samples + quantile[0] - 1) // quantile[0]
    nas_min = (num_individual_samples + quantile[-1] - 1) // quantile[-1]

    # reset num_aggregated_samples
    num_aggregated_samples = len(random_offsets) - 1

    # check num_aggregated_samples
    if num_aggregated_samples < nas_min or nas_max < num_aggregated_samples:
        raise ValueError("Sannity check failed")

    sample_partition = [0]
    for l in range(num_aggregated_samples):
        #s = random_offsets[l]
        e = random_offsets[l + 1]
        sample_partition.append(e)

    np.save(os.path.join(dump_path, "sample_partition.npy"), np.array(sample_partition, dtype=np.int32))


tar_fea = 1   # single target
den_fea = 13  # 13 dense  features
spa_fea = 26  # 26 sparse features
tot_fea = tar_fea + den_fea + spa_fea

data_file = open(os.path.join(data_path, 'terabyte_processed_test.bin'), 'rb')
raw_data = data_file.read()
array = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, tot_fea)

x_int_batch = array[:, 1:14]
x_cat_batch = array[:, 14:]
y_batch = array[:, 0].reshape(-1)

np.save(os.path.join(output_path, "x_int_batch.npy"), x_int_batch)
np.save(os.path.join(output_path, "x_cat_batch.npy"), x_cat_batch)
np.save(os.path.join(output_path, "y_batch.npy"), y_batch)

# process feature count file
day_day_count = np.load(os.path.join(data_path, 'day_day_count.npz'))['total_per_file']
day_fea_count = np.load(os.path.join(data_path, 'day_fea_count.npz'))['counts']
np.save(os.path.join(output_path, 'day_day_count.npy'), day_day_count)
np.save(os.path.join(output_path, 'day_fea_count.npy'), day_fea_count)

generate_sample_partition(data_path, output_path, "./dist_quantile.txt")
