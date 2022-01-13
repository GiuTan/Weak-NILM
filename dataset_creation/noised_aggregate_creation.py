import random as python_random
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser(description="Noised aggregate creation")
parser.add_argument("--building", type=int, default=2, help="Desired building")
parser.add_argument("--num_of_bags",type=int, default=2000, help="Number of bags created for the desired building")
parser.add_argument("--noise_path", type=str, default='', help="Path where noise has been saved")
parser.add_argument("--agg_synth_path", type=str, default='', help="Path where synth aggregate has been saved")
parser.add_argument("--agg_noised_path", type=str, default='', help="Path where noised aggregate has been saved")
arguments = parser.parse_args()

def noise_segmentation(k, noise_path,num_of_bag):

    vector_noise = np.load(noise_path + 'noise_'+str(k)+'.npy', allow_pickle=True)
    print("shape noise", vector_noise.shape)
    print("shape noise", vector_noise[0].shape)
    results = []
    results = np.array(results)
    for lung in range(len(vector_noise)):
        results = np.concatenate([results,vector_noise[lung]], axis=0)
    print("Shape total vector:", results.shape)

    random_list = random.sample(range(0, (len(results) - 2550)), (len(results) - 2550))
    print("index control 1")
    print(random_list[0])
    vector_list_1 = []
    for i in random_list[:num_of_bag]:
        vector = results[i: (i + 2550)]
        vector_list_1.append(vector)

    return np.array(vector_list_1)

if __name__ == '__main__':
    np.random.seed(123)
    python_random.seed(123)


    agg_synth_path = arguments.agg_synth_path
    agg_noised_path = arguments.agg_noised_path
    building = arguments.building
    noise_path = arguments.noise_path
    noise = noise_segmentation(building, noise_path, arguments.num_of_bags)
    len_noise = len(noise)
    print(len_noise)
    print(noise.shape)
    n_= 0
    k = arguments.building
    for i in range(arguments.num_of_bags):
            try:
                agg = np.load(agg_synth_path + 'house_' + str(k) + '/aggregate_%d.npy' %i)
            except FileNotFoundError:
                continue
            noise_ = np.nan_to_num(noise[n_], nan=1)
            agg = np.add(noise_, agg)
            n_ += 1
            np.save(agg_noised_path + "house_" + str(k) + "/aggregate_%d" % i,agg)

    print("total len noise", len(noise))


