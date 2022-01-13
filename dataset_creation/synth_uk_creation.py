import numpy as np
from os.path import join
import pandas as pd
from nilmtk.nilmtk.dataset import DataSet
from nilmtk.nilmtk.electric import Electric
import json
import argparse
import gc
#from uk_appliance_info import *
np.random.seed(0) # for strong
#np.random.seed(3) # for weak
parser = argparse.ArgumentParser(description="UK-DALE synthetic dataset creation")

parser.add_argument("--tot_pos_sample_1", type=int, default=1400000, help="Number of total positive samples from house 1")
parser.add_argument("--tot_pos_sample_2", type=int, default=120000, help="Number of total positive samples from house 2")
parser.add_argument("--tot_pos_sample_3", type=int, default=200000, help="Number of total positive samples from house 3")
parser.add_argument("--tot_pos_sample_4", type=int, default=100000, help="Number of total positive samples from house 4")
parser.add_argument("--tot_pos_sample_5", type=int, default=93350, help="Number of total positive samples from house 5")
parser.add_argument("--building", type= int, default=1, help="House for bags creation")
parser.add_argument("--window_length", type= int, default=2550, help="Segments dimension")
parser.add_argument("--number_shift", type= int, default=250, help="Number of samples from which randomly select activations shifting")
arguments = parser.parse_args()


min_off_duration = {
    'washing machine': 30,
    'dish washer': 1800,
    'microwave': 30,
    'kettle': 0,
    'fridge': 12
}

min_on_duration = {
    'washing machine': 1800,
    'dish washer': 1800,
    'microwave': 12,
    'kettle': 12,
    'fridge': 60
}

on_power_threshold = {
    'washing machine': 20,
    'dish washer': 10,
    'microwave': 200,
    'kettle': 2000,
    'fridge': 50

}

activations_1 = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}

activations_3 = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}

activations_4 = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}


activations_2 = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}

activations_5 = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}

indices_of_activations = {
    'washing machine': [],
    'kettle': [],
    'fridge': [],
    'dish washer': [],
    'microwave': []
}


#    SAVE THE DATA AND LABELS   #
dict_1 = {'labels_0': {
    'strong':[],
    'weak':[],
}}

dict_2 = {'labels_0': {
    'strong':[],
    'weak':[],
}}

dict_5 = {'labels_0': {
    'strong':[],
    'weak':[],
}}

dict_3 = {'labels_0': {
    'strong':[],
    'weak':[],
}}

dict_4 = {'labels_0': {
    'strong':[],
    'weak':[],
}}


destination_path = "../aggregate_data/"
ukdale_path = "../ukdale.h5"
ukdale = DataSet(ukdale_path)
if arguments.building == 3:
    appliances = ['kettle','','','','']
if arguments.building == 4:
    appliances = ['','','fridge','','']
if arguments.building == 1 or arguments.building == 2 or arguments.building == 5:
    appliances = ['kettle', 'microwave','fridge', 'washing machine', 'dish washer']

phase_change = 5 # number of appliances to which subdivide total positive samples
samples_per_class_1 = round((arguments.tot_pos_sample_1 / phase_change))
samples_per_class_5 = round((arguments.tot_pos_sample_5 / phase_change))
samples_per_class_2 = round((arguments.tot_pos_sample_2 / phase_change))
samples_per_class_3 = arguments.tot_pos_sample_3
samples_per_class_4 = arguments.tot_pos_sample_4
window_length = 2550
number_shift = arguments.number_shift
cases = 5
list_kettle = []
list_micro = []
list_fridge = []
list_wash = []
list_dish = []

def activation_appliances_nilmtk(appliances, building):
    print("Extracting activations")
    for appliance in appliances:
            elec = ukdale.buildings[building].elec
            curr_appliance = elec[appliance]
            activation_ = Electric.get_activations(curr_appliance, min_off_duration=min_off_duration[appliance],
                                                   min_on_duration=min_on_duration[appliance],
                                                   on_power_threshold=on_power_threshold[appliance])
            df = next(curr_appliance.load(ac_type='active'))
            df.head()
            print(building)
            print(appliance)
            print(len(activation_))
            if building == 1:
                activations_1[
                    appliance] = activation_

            if building == 2:
                activations_2[appliance] = activation_

            if building == 5:
                activations_5[appliance] = activation_

            if building == 3:
                activations_3[appliance] = activation_

            if building == 4:
                activations_4[appliance] = activation_
    if building == 1:
        return activations_1
    if building == 2:
        return activations_2
    if building == 3:
        return activations_3
    if building == 4:
        return activations_4
    if building ==5:
        return activations_5



def indexes_annotation(app1,app2,app3,app4,app5, rand_1, rand_2,rand_3, rand_4, rand_5):


    if app1 == 'kettle':
        list_kettle.append(rand_1)

    elif app2 == 'kettle':
        list_kettle.append(rand_2)

    elif app3 == 'kettle':
        list_kettle.append(rand_3)

    elif app4 == 'kettle':
        list_kettle.append(rand_4)

    elif app5 == 'kettle':
        list_kettle.append(rand_5)


    if app1 == 'microwave':
        list_micro.append(rand_1)

    elif app2 == 'microwave':
        list_micro.append(rand_2)

    elif app3 == 'microwave':
        list_micro.append(rand_3)

    elif app4 == 'microwave':
        list_micro.append(rand_4)

    elif app5 == 'microwave':
        list_micro.append(rand_5)


    if app1 == 'fridge':
        list_fridge.append(rand_1)

    elif app2 == 'fridge':
        list_fridge.append(rand_2)

    elif app3 == 'fridge':
        list_fridge.append(rand_3)

    elif app4 == 'fridge':
        list_fridge.append(rand_4)

    elif app5 == 'fridge':
        list_fridge.append(rand_5)


    if app1 == 'washing machine':
        list_wash.append(rand_1)

    elif app2 == 'washing machine':
        list_wash.append(rand_2)

    elif app3 == 'washing machine':
        list_wash.append(rand_3)

    elif app4 == 'washing machine':
        list_wash.append(rand_4)

    elif app5 == 'washing machine':
        list_wash.append(rand_5)


    if app1 == 'dish washer':
        list_dish.append(rand_1)

    elif app2 == 'dish washer':
        list_dish.append(rand_2)

    elif app3 == 'dish washer':
        list_dish.append(rand_3)

    elif app4 == 'dish washer':
        list_dish.append(rand_4)

    elif app5 == 'dish washer':
        list_dish.append(rand_5)


def sample_counter(app1,app2,app3,app4,app5,sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d, act1,act2,act3,act4,act5):


    if app1 == 'kettle':
        sample_count_k += len(act1)

    if app2 == 'kettle':
        sample_count_k += len(act2)

    if app3 == 'kettle':
        sample_count_k += len(act3)

    if app4 == 'kettle':
        sample_count_k += len(act4)

    if app5 == 'kettle':
        sample_count_k += len(act5)


    if app1 == 'microwave':
        sample_count_m += len(act1)

    if app2 == 'microwave':
        sample_count_m += len(act2)

    if app3 == 'microwave':
        sample_count_m += len(act3)

    if app4 == 'microwave':
        sample_count_m += len(act4)

    if app5 == 'microwave':
        sample_count_m += len(act5)


    if app1 == 'fridge':
        sample_count_f += len(act1)

    if app2 == 'fridge':
        sample_count_f += len(act2)

    if app3 == 'fridge':
        sample_count_f += len(act3)

    if app4 == 'fridge':
        sample_count_f += len(act4)

    if app5 == 'fridge':
        sample_count_f += len(act5)


    if app1 == 'washing machine':
        sample_count_w += len(act1)

    if app2 == 'washing machine':
        sample_count_w += len(act2)

    if app3 == 'washing machine':
        sample_count_w += len(act3)

    if app4 == 'washing machine':
        sample_count_w += len(act4)

    if app5 == 'washing machine':
        sample_count_w += len(act5)

    if app1 == 'dish washer':
        sample_count_d += len(act1)

    elif app2 == 'dish washer':
        sample_count_d += len(act2)

    if app3 == 'dish washer':
        sample_count_d += len(act3)

    if app4 == 'dish washer':
        sample_count_d += len(act4)

    if app5 == 'dish washer':
        sample_count_d += len(act5)

    return sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d

def repetition_counter(appliances):

    indexes_k = []
    indexes_m = []
    indexes_f = []
    indexes_w = []
    indexes_d = []
    list_indexes = [indexes_k, indexes_m, indexes_f, indexes_w, indexes_d]
    for i in range(len(appliances)):
        indexes = list_indexes[i]
        if appliances[i] != '':
            for idx in indices_of_activations[appliances[i]]:
                repetition = indices_of_activations[appliances[i]].count(idx)
                indexes.append(repetition)

    return indexes_k, indexes_m, indexes_f, indexes_w, indexes_d

def padd_shift(rand_shift,app):

    padd = pd.Series(np.zeros(rand_shift))
    app_series = app['power']['active']
    app_series = app_series.append(padd, ignore_index=True).shift(periods=rand_shift, fill_value=0)
    return app_series

def strong_labels_creation(appliance1,app1,appliance2, app2,appliance3, app3, appliance4, app4, appliance5, app5, flag = 0):

    list_vectors_strong = []
    vector_strong_1 = pd.Series()
    vector_strong_2 = pd.Series()
    vector_strong_3 = pd.Series()
    vector_strong_4 = pd.Series()
    vector_strong_5 = pd.Series()

    if flag == 4:
            vector_strong_1 = pd.Series()

            list_1 = []
            if appliance1 == 'kettle':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['kettle']:
                        list_1.append(0)
                    else:
                        list_1.append(1)
                print("Appliance 1 vector strong Kettle")
                vector_strong_1_a = pd.Series(list_1)
            else:
                vector_strong_1_a = pd.Series(np.zeros(window_length))

            list_1 = []
            if appliance2 == 'kettle':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['kettle']:
                        list_1.append(0)
                    else:
                        list_1.append(1)

                print("Appliance 2 vector strong Kettle")
                vector_strong_1_b = pd.Series(list_1)
            else:
                vector_strong_1_b = pd.Series(np.zeros(window_length))

            list_1 = []
            if appliance3 == 'kettle':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['kettle']:
                        list_1.append(0)
                    else:
                        list_1.append(1)

                print("Appliance 3 vector strong Kettle")
                vector_strong_1_c = pd.Series(list_1)
            else:
                vector_strong_1_c = pd.Series(np.zeros(window_length))

            list_1 = []
            if appliance4 == 'kettle':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['kettle']:
                        list_1.append(0)
                    else:
                        list_1.append(1)

                print("Appliance 4 vector strong Kettle")
                vector_strong_1_d = pd.Series(list_1)
            else:
                vector_strong_1_d = pd.Series(np.zeros(window_length))

            list_1 = []
            if appliance5 == 'kettle':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['kettle']:
                        list_1.append(0)
                    else:
                        list_1.append(1)

                print("Appliance 5 vector strong Kettle")
                vector_strong_1_e = pd.Series(list_1)
            else:
                vector_strong_1_e = pd.Series(np.zeros(window_length))


            vector_strong_1 = vector_strong_1_a.add(vector_strong_1_b)
            vector_strong_1 = vector_strong_1.add(vector_strong_1_c)
            vector_strong_1 = vector_strong_1.add(vector_strong_1_d)
            vector_strong_1 = vector_strong_1.add(vector_strong_1_e)

            list_2 = []
            vector_strong_2 = pd.Series()
            if appliance1 == 'microwave':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)
                vector_strong_2_a = pd.Series(list_2)
            else:
                vector_strong_2_a = pd.Series(np.zeros(window_length))

            list_2 = []
            if appliance2 == 'microwave':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)
                vector_strong_2_b = pd.Series(list_2)
            else:
                vector_strong_2_b = pd.Series(np.zeros(window_length))

            list_2 = []
            if appliance3 == 'microwave':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)

                print("Appliance 3 vector strong microwave")
                vector_strong_2_c = pd.Series(list_2)
            else:
                vector_strong_2_c = pd.Series(np.zeros(window_length))

            list_2 = []
            if appliance4 == 'microwave':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)

                print("Appliance 4 vector strong microwave")
                vector_strong_2_d = pd.Series(list_2)
            else:
                vector_strong_2_d = pd.Series(np.zeros(window_length))

            list_2 = []
            if appliance5 == 'microwave':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)

                print("Appliance 5 vector strong microwave")
                vector_strong_2_e = pd.Series(list_2)
            else:
                vector_strong_2_e = pd.Series(np.zeros(window_length))

            vector_strong_2 = vector_strong_2_a.add(vector_strong_2_b)
            vector_strong_2 = vector_strong_2.add(vector_strong_2_c)
            vector_strong_2 = vector_strong_2.add(vector_strong_2_d)
            vector_strong_2 = vector_strong_2.add(vector_strong_2_e)


            list_3 = []
            vector_strong_3 = pd.Series()
            if appliance1 == 'fridge':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)
                vector_strong_3_a = pd.Series(list_3)
            else:
                vector_strong_3_a = pd.Series(np.zeros(window_length))

            list_3 = []
            if appliance2 == 'fridge':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)
                vector_strong_3_b = pd.Series(list_3)
            else:
                vector_strong_3_b = pd.Series(np.zeros(window_length))

            list_3 = []
            if appliance3 == 'fridge':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)

                print("Appliance 3 vector strong microwave")
                vector_strong_3_c = pd.Series(list_3)
            else:
                vector_strong_3_c = pd.Series(np.zeros(window_length))

            list_3 = []
            if appliance4 == 'fridge':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)

                print("Appliance 4 vector strong microwave")
                vector_strong_3_d = pd.Series(list_3)
            else:
                vector_strong_3_d = pd.Series(np.zeros(window_length))

            list_3 = []
            if appliance5 == 'fridge':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)

                print("Appliance 5 vector strong microwave")
                vector_strong_3_e = pd.Series(list_3)
            else:
                vector_strong_3_e = pd.Series(np.zeros(window_length))

            vector_strong_3 = vector_strong_3_a.add(vector_strong_3_b)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_c)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_d)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_e)




            list_4 = []
            vector_strong_4 = pd.Series()
            if appliance1 == 'washing machine':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['washing machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)
                vector_strong_4 = pd.Series(list_4)
            elif appliance2 == 'washing machine':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['washing machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)
                vector_strong_4 = pd.Series(list_4)


            elif appliance3 == 'washing machine':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['washing machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 3 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)

            elif appliance4 == 'washing machine':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['washing machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 4 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)
            elif appliance5 == 'washing machine':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['washing machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 5 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)
            else:
                vector_strong_4 = pd.Series(np.zeros(window_length))

            list_5 = []
            vector_strong_5 = pd.Series()
            if appliance1 == 'dish washer':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['dish washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)
                vector_strong_5 = pd.Series(list_5)

            elif appliance2 == 'dish washer':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['dish washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)
                vector_strong_5 = pd.Series(list_5)

            elif appliance3 == 'dish washer':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['dish washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 3 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            elif appliance4 == 'dish washer':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['dish washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 4 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            elif appliance5 == 'dish washer':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['dish washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 5 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            else:
                vector_strong_5 = pd.Series(np.zeros(window_length))




    list_vectors_strong.append(vector_strong_1)
    list_vectors_strong.append(vector_strong_2)
    list_vectors_strong.append(vector_strong_3)
    list_vectors_strong.append(vector_strong_4)
    list_vectors_strong.append(vector_strong_5)

    return list_vectors_strong

def data_iteration(activations_, samples_per_class, building):

    final_weak = []
    final_strong = []
    aggregate_ = []

    if building == 4 or building == 3:
        sample_count_k = 0
        sample_count_m = 0
        sample_count_f = 0
        sample_count_w = 0
        sample_count_d = 0

        for a in range(10000):  # l'AGGREGATO è LA SOMMA DI 1 APPLIANCE
            if not (
                    sample_count_f > samples_per_class or sample_count_k > samples_per_class ):

                if building == 4:
                    i = 2
                else:
                    i = 0
                mu, sigma = 0, 1
                s = np.random.normal(mu, sigma, window_length)
                s = pd.Series(s)
                s_app = pd.Series(np.zeros(window_length))

                print("Appliance 1")
                print(appliances[i])

                if (appliances[i] == 'kettle' and sample_count_k >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]


                rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)

                app1 = activations_[appliances[i]][rand_1[0]]
                app1 = app1.reset_index()
                app1 = app1.drop(["index"], axis=1)

                if appliances[i] == 'kettle':
                    rand_shift = np.random.randint((number_shift + 700), size=1)
                    rand_shift_mkf1 = np.random.randint((rand_shift[0] + 200 - 100), size=1)
                    rand_shift_mkf2 = np.random.randint((rand_shift[0] + 500), size=1)
                    rand_shift_mkf3 = np.random.randint((rand_shift[0] + 300), size=1)
                    app_mkf1 = activations_[appliances[i]][rand_1[0] - 1]
                    app_mkf1 = app_mkf1.reset_index()
                    app_mkf1 = app_mkf1.drop(["index"], axis=1)

                    app_mkf2 = activations_[appliances[i]][rand_1[0] - 5]
                    app_mkf2 = app_mkf2.reset_index()
                    app_mkf2 = app_mkf2.drop(["index"], axis=1)

                    app_mkf3 = activations_[appliances[i]][rand_1[0] - 7]
                    app_mkf3 = app_mkf3.reset_index()
                    app_mkf3 = app_mkf3.drop(["index"], axis=1)

                    len1 = len(app_mkf1['power']['active'])
                    len2 = len(app_mkf2['power']['active'])
                    len3 = len(app_mkf3['power']['active'])

                    app_mkf1 = padd_shift(rand_shift_mkf1[0], app_mkf1)

                    app_mkf2 = padd_shift(rand_shift_mkf2[0], app_mkf2)

                    app_mkf3 = padd_shift(rand_shift_mkf3[0], app_mkf3)

                    if appliances[i] == 'kettle':
                        sample_count_k = sample_count_k + len1 + len2 + len3
                    elif appliances[i] == 'microwave':
                        sample_count_m = sample_count_m + len1 + len2 + len3
                    elif appliances[i] == 'fridge':
                        sample_count_f = sample_count_f + len1 + len2 + len3

                    app1_series = padd_shift(rand_shift[0], app1)
                    app1_series = app1_series.add(app_mkf1, fill_value=0)
                    app1_series = app1_series.add(app_mkf2, fill_value=0)
                    app1_series = app1_series.add(app_mkf3, fill_value=0)

                else:
                    rand_shift = np.random.randint(number_shift, size=1)
                    app1_series = padd_shift(rand_shift[0], app1)

                aggregate = app1_series.add(s, fill_value=0)
                aggregate = aggregate.to_numpy()
                aggregate[aggregate < 0] = 0
                aggregate = pd.Series(aggregate)
                aggregate_.append(aggregate)
                app1_series = app1_series.add(s_app, fill_value=0)

                sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(
                    appliances[i], 0, 0, 0,
                    0,
                    sample_count_k, sample_count_m,
                    sample_count_f, sample_count_w, sample_count_d, app1['power']['active'], 0,
                    0, 0, 0)

                list_vectors_strong = strong_labels_creation(appliance1=appliances[i], app1=app1_series, appliance2=0,
                                                             app2=0,
                                                             appliance3=0, app3=0, appliance4=0, app4=0, appliance5=0,
                                                             app5=0, flag=4)
                final_strong.append(list_vectors_strong)
                vector_weak = [0, 0, 0, 0, 0]
                vector_weak[i] = 1
                print(vector_weak)
                final_weak.append(vector_weak)
                # print(list_indices)
                indexes_annotation(appliances[i], 0, 0, 0, 0, rand_1[0], 0, 0, 0, 0)
            else:
                break

    else:
        print("Aggregate with one appliance")
        sample_count_k = 0
        sample_count_m = 0
        sample_count_f = 0
        sample_count_w = 0
        sample_count_d = 0

        for a in range(10000):
            if not (sample_count_k > samples_per_class and sample_count_m > samples_per_class and sample_count_w > samples_per_class and
                    sample_count_f > samples_per_class and sample_count_d > samples_per_class):

                i = np.random.randint(len(appliances), size=1)
                i = i[0]
                mu, sigma = 0, 1
                s = np.random.normal(mu, sigma, window_length)
                s = pd.Series(s)
                s_app = pd.Series(np.zeros(window_length))



                if (appliances[i] == 'kettle' and sample_count_k >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]

                rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)

                app1 = activations_[appliances[i]][rand_1[0]]
                app1 = app1.reset_index()
                app1 = app1.drop(["index"], axis=1)

                if appliances[i] == 'kettle':
                    rand_shift = np.random.randint((number_shift + 700), size=1)
                    rand_shift_mkf1 = np.random.randint((rand_shift[0] + 200 - 100), size=1)
                    rand_shift_mkf2 = np.random.randint((rand_shift[0] + 500), size=1)
                    rand_shift_mkf3 = np.random.randint((rand_shift[0] + 300), size=1)
                    app_mkf1 = activations_[appliances[i]][rand_1[0] - 1]
                    app_mkf1 = app_mkf1.reset_index()
                    app_mkf1 = app_mkf1.drop(["index"], axis=1)

                    app_mkf2 = activations_[appliances[i]][rand_1[0] - 5]
                    app_mkf2 = app_mkf2.reset_index()
                    app_mkf2 = app_mkf2.drop(["index"], axis=1)

                    app_mkf3 = activations_[appliances[i]][rand_1[0] - 7]
                    app_mkf3 = app_mkf3.reset_index()
                    app_mkf3 = app_mkf3.drop(["index"], axis=1)

                    len1 = len(app_mkf1['power']['active'])
                    len2 = len(app_mkf2['power']['active'])
                    len3 = len(app_mkf3['power']['active'])

                    app_mkf1 = padd_shift(rand_shift_mkf1[0], app_mkf1)

                    app_mkf2 = padd_shift(rand_shift_mkf2[0], app_mkf2)

                    app_mkf3 = padd_shift(rand_shift_mkf3[0], app_mkf3)

                    if appliances[i] == 'kettle':
                        sample_count_k = sample_count_k + len1 + len2 + len3
                    elif appliances[i] == 'microwave':
                        sample_count_m = sample_count_m + len1 + len2 + len3
                    elif appliances[i] == 'fridge':
                        sample_count_f = sample_count_f + len1 + len2 + len3

                    app1_series = padd_shift(rand_shift[0], app1)
                    app1_series = app1_series.add(app_mkf1, fill_value=0)
                    app1_series = app1_series.add(app_mkf2, fill_value=0)
                    app1_series = app1_series.add(app_mkf3, fill_value=0)

                else:
                    rand_shift = np.random.randint(number_shift, size=1)
                    app1_series = padd_shift(rand_shift[0], app1)

                aggregate = app1_series.add(s, fill_value=0)
                aggregate = aggregate.to_numpy()
                aggregate[aggregate < 0] = 0
                aggregate = pd.Series(aggregate)
                aggregate_.append(aggregate)
                app1_series = app1_series.add(s_app, fill_value=0)

                sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(
                    appliances[i], 0, 0, 0,
                    0,
                    sample_count_k, sample_count_m,
                    sample_count_f, sample_count_w, sample_count_d, app1['power']['active'], 0,
                    0, 0, 0)

                list_vectors_strong = strong_labels_creation(appliance1=appliances[i], app1=app1_series, appliance2=0,
                                                             app2=0,
                                                             appliance3=0, app3=0, appliance4=0, app4=0, appliance5=0,
                                                             app5=0, flag=4)
                final_strong.append(list_vectors_strong)
                vector_weak = [0, 0, 0, 0, 0]
                vector_weak[i] = 1
                print(vector_weak)
                final_weak.append(vector_weak)
                # print(list_indices)
                indexes_annotation(appliances[i], 0, 0, 0, 0, rand_1[0], 0, 0, 0, 0)
            else:
                break
                # print("Sequences with 1 appliances:")
                # print(a)
        sample_count_k = 0
        sample_count_m = 0
        sample_count_f = 0
        sample_count_w = 0
        sample_count_d = 0

        # Aggregate is the sum of two appliances
        print("Aggregate with two appliances")
        for a in range(125000):
            if not (sample_count_k > samples_per_class and sample_count_m > samples_per_class and sample_count_f > samples_per_class and sample_count_w > samples_per_class and sample_count_d > samples_per_class):


                i = np.random.randint(len(appliances), size=1)
                i = i[0]
                list_indices_app1 = []
                list_indices_app2 = []

                rand_app = np.random.randint(len(appliances), size=1)
                mu, sigma = 0, 1
                s = np.random.normal(mu, sigma,window_length)
                s = pd.Series(s)
                s_app = pd.Series(np.zeros(window_length))



                if (appliances[i] =='kettle' and sample_count_k >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]
                elif (appliances[i] == 'microwave' and sample_count_m >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]
                elif (appliances[i] == 'fridge' and sample_count_f >= samples_per_class):
                    # i = np.random.randint(len(appliances) - 3, size=1)
                    # i = i[0]
                    i = 1
                elif (appliances[i] == 'washing machine' and sample_count_w >= samples_per_class):
                    #i = np.random.randint(len(appliances) - 3, size=1)
                    #i = i[0]
                    i = 1
                elif (appliances[i] == 'dish washer' and sample_count_d > samples_per_class):
                    # i = np.random.randint(len(appliances) - 3, size=1)
                    # i = i[0]
                    i = 1

                if rand_app[0] == i:
                        rand_app = np.random.randint(5, size=1)
                if rand_app[0] == i:
                        rand_app = np.random.randint(5, size=1)



                if (appliances[rand_app[0]] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) , size=1)
                elif (appliances[rand_app[0]] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app = np.random.randint(len(appliances), size=1)
                elif (appliances[rand_app[0]] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                elif (appliances[rand_app[0]] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                elif (appliances[rand_app[0]] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3 , size=1)

                rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
                list_indices_app1.append(rand_1)
                app1 = activations_[appliances[i]][rand_1[0]]
                app1 = app1.reset_index()
                app1 = app1.drop(["index"], axis=1)
                app1_series = app1['power']['active']

                rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
                list_indices_app2.append(rand_2)
                app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
                app2 = app2.reset_index()
                app2 = app2.drop(["index"], axis=1)

                sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(appliances[i], appliances[rand_app[0]], 0, 0, 0, sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d, app1_series, app2['power']['active'], 0,0,0)



                app1_series = app1_series.add(s_app, fill_value=0)

                rand_shift = np.random.randint(number_shift, size=1)
                app2_series = padd_shift(rand_shift[0], app2)
                app2_series = app2_series.add(s_app, fill_value=0)

                vector = app2_series.add(app1_series, fill_value=0)
                aggregate = vector.add(s, fill_value=0)
                aggregate = aggregate.to_numpy()
                aggregate[aggregate < 0] = 0
                aggregate = pd.Series(aggregate)
                aggregate_.append(aggregate)

                list_vectors_strong = strong_labels_creation(appliance1=appliances[i], app1=app1_series,
                                                                     appliance2=appliances[rand_app[0]], app2=app2_series,
                                                                     appliance3=0, app3=0, appliance4=0, app4=0, appliance5=0,
                                                                     app5=0, flag=4)
                final_strong.append(list_vectors_strong)
                vector_weak = [0, 0, 0, 0, 0]
                vector_weak[i] = 1
                vector_weak[rand_app[0]] = 1
                print(vector_weak)
                final_weak.append(vector_weak)
                indexes_annotation(appliances[i], appliances[rand_app[0]], 0, 0, 0, rand_1[0],
                                           rand_2[0], 0, 0, 0)

            else:
                break

        sample_count_k = 0
        sample_count_m = 0
        sample_count_f = 0
        sample_count_w = 0
        sample_count_d = 0

        print("Aggregate with three appliances")
        for a in range(125000):
            if not (
                    sample_count_k > samples_per_class and sample_count_m > samples_per_class and sample_count_f > samples_per_class and sample_count_w > samples_per_class and sample_count_d > samples_per_class):

                i = np.random.randint(len(appliances), size=1)
                i = i[0]

                rand_app = np.random.randint(len(appliances), size=1)
                rand_app2 = 0
                mu, sigma = 0, 1
                s = np.random.normal(mu, sigma, window_length)
                s = pd.Series(s)
                s_app = pd.Series(np.zeros(window_length))



                if (appliances[i] =='kettle' and sample_count_k >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]
                elif (appliances[i] == 'microwave' and sample_count_m >= samples_per_class):
                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]
                elif (appliances[i] == 'fridge' and sample_count_f >= samples_per_class):
                    i = np.random.randint(len(appliances) - 3, size=1)
                    i = i[0]
                elif (appliances[i] == 'washing machine' and sample_count_w >= samples_per_class):
                    i = np.random.randint(len(appliances) - 3, size=1)
                    i = i[0]
                elif (appliances[i] == 'dish washer' and sample_count_d > samples_per_class):
                    i = np.random.randint(len(appliances) - 3, size=1)
                    i = i[0]


                rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
                app1 = activations_[appliances[i]][rand_1[0]]
                app1 = app1.reset_index()
                app1 = app1.drop(["index"], axis=1)
                app1_series = app1['power']['active']
                app1_series = app1_series.add(s_app, fill_value=0)
                if rand_app[0] == i:
                        rand_app = np.random.randint(cases, size=1)
                if rand_app[0] == i:
                        rand_app = np.random.randint(cases, size=1)
                if i != 0 and rand_app[0] != 0:
                        rand_app2 = 0
                if i != 1 and rand_app[0] != 1:
                        rand_app2 = 1
                if i != 2 and rand_app[0] != 2:
                        rand_app2 = 2
                if i != 3 and rand_app[0] != 3 and i != 4 and rand_app[0] != 4:
                        rand_app2 = 3
                if i != 4 and rand_app[0] != 4 and i!= 3 and rand_app[0] != 3:
                        rand_app2 = 4



                if (appliances[rand_app[0]] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                elif (appliances[rand_app[0]] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app = np.random.randint(len(appliances), size=1)
                elif (appliances[rand_app[0]] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 1, size=1)
                elif (appliances[rand_app[0]] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                elif (appliances[rand_app[0]] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 2 , size=1)



                if (appliances[rand_app2] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app2 = 1
                elif (appliances[rand_app2] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app2 = 0
                elif (appliances[rand_app2] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app2 = 0
                elif (appliances[rand_app2] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app2 = 1
                elif (appliances[rand_app2] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app2 = 1


                rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
                app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
                app2 = app2.reset_index()
                app2 = app2.drop(["index"], axis=1)

                rand_shift = np.random.randint(number_shift, size=1)
                app2_series = padd_shift(rand_shift[0], app2)
                app2_series = app2_series.add(s_app, fill_value=0)

                rand_3 = np.random.randint(len(activations_[appliances[rand_app2]]), size=1)
                app3 = activations_[appliances[rand_app2]][rand_3[0]]
                app3 = app3.reset_index()
                app3 = app3.drop(["index"], axis=1)

                sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(appliances[i], appliances[rand_app[0]], appliances[rand_app2], 0, 0, sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d, app1['power']['active'], app2['power']['active'], app3['power']['active'],0,0)

                rand_shift = np.random.randint(number_shift, size=1)
                app3_series = padd_shift(rand_shift[0], app3)
                app3_series = app3_series.add(s_app, fill_value=0)

                vector = app2_series.add(app1_series, fill_value=0)
                vector = vector.add(app3_series, fill_value=0)
                aggregate = vector.add(s, fill_value=0)
                aggregate = aggregate.to_numpy()
                aggregate[aggregate < 0] = 0
                aggregate = pd.Series(aggregate)
                aggregate_.append(aggregate)


                list_vectors_strong = strong_labels_creation(appliance1=appliances[i], app1=app1_series,
                                                                 appliance2=appliances[rand_app[0]], app2=app2_series,
                                                                 appliance3=appliances[rand_app2], app3=app3_series, appliance4=0,
                                                                 app4=0, appliance5=0, app5=0, flag=4)
                final_strong.append(list_vectors_strong)
                vector_weak = [0, 0, 0, 0, 0]
                vector_weak[i] = 1
                vector_weak[rand_app[0]] = 1
                vector_weak[rand_app2] = 1
                print(vector_weak)
                final_weak.append(vector_weak)

                indexes_annotation(appliances[i], appliances[rand_app[0]], appliances[rand_app2], 0, 0, rand_1[0], rand_2[0], rand_3[0], 0, 0)

            else:
                break

        sample_count_k = 0
        sample_count_m = 0
        sample_count_f = 0
        sample_count_w = 0
        sample_count_d = 0

        # Aggregate is the sum of 4 appliances

        for a in range(125000):
            if not (
                    sample_count_k > samples_per_class and sample_count_m > samples_per_class and sample_count_f > samples_per_class and sample_count_w > samples_per_class and sample_count_d > samples_per_class):

                    i = np.random.randint(len(appliances), size=1)
                    i = i[0]
                    rand_app = np.random.randint(len(appliances), size=1)
                    rand_app2 = 0
                    rand_app3 = 0
                    mu, sigma = 0, 1
                    s = np.random.normal(mu, sigma, window_length)
                    s = pd.Series(s)
                    s_app = pd.Series(np.zeros(window_length))

                    print("Appliance 1")
                    print(appliances[i])

                    if (appliances[i] == 'kettle' and sample_count_k >= samples_per_class):
                        i = np.random.randint(len(appliances) - 2 , size=1)
                        i = i[0]
                    elif (appliances[i] == 'microwave' and sample_count_m >= samples_per_class):
                        i = np.random.randint(len(appliances) - 1, size=1)
                        i = i[0]
                    elif (appliances[i] == 'fridge' and sample_count_f >= samples_per_class):
                        i = np.random.randint(len(appliances) - 3, size=1)
                        i = i[0]
                    elif (appliances[i] == 'washing machine' and sample_count_w >= samples_per_class):
                        i = np.random.randint(len(appliances) - 2, size=1)
                        i = i[0]
                    elif (appliances[i] == 'dish washer' and sample_count_d > samples_per_class):
                        i = np.random.randint(len(appliances) - 2, size=1)
                        i = i[0]


                    rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
                    app1 = activations_[appliances[i]][rand_1[0]]
                    app1 = app1.reset_index()
                    app1 = app1.drop(["index"], axis=1)
                    app1_series = app1['power']['active']
                    app1_series = app1_series.add(s_app, fill_value=0)
                    if rand_app[0] == i:
                        rand_app = np.random.randint(cases, size=1)
                    if rand_app[0] == i:
                        rand_app = np.random.randint(cases, size=1)
                    if i != 0 and rand_app[0] != 0:
                        rand_app2 = 0
                    if i != 1 and rand_app[0] != 1:
                        rand_app2 = 1
                    if i != 2 and rand_app[0] != 2:
                        rand_app2 = 2
                    if i != 3 and rand_app[0] != 3 and i != 4 and rand_app[0] != 4:
                        rand_app2 = 3
                    if i != 4 and rand_app[0] != 4 and i != 3 and rand_app[0] != 3:
                        rand_app2 = 4
                    if i != 0 and rand_app[0] != 0 and rand_app2 != 0:
                        rand_app3 = 0
                    if i != 1 and rand_app[0] != 1 and rand_app2 != 1:
                        rand_app3 = 1
                    if i != 2 and rand_app[0] != 2 and rand_app2 != 2 and i != 4 and rand_app[0] != 4 and rand_app2 != 4:
                        rand_app3 = 2
                    if i != 3 and rand_app[0] != 3 and rand_app2 != 3 and i != 4 and rand_app[0] != 4 and rand_app2 != 4:
                        rand_app3 = 3
                    if i != 4 and rand_app[0] != 4 and rand_app2 != 4 and i!=3 and rand_app[0] != 3 and rand_app2 !=3:
                        rand_app3 = 4



                    if (appliances[rand_app[0]] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 2, size=1)
                    elif (appliances[rand_app[0]] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app = np.random.randint(len(appliances), size=1)
                    elif (appliances[rand_app[0]] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                    elif (appliances[rand_app[0]] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)
                    elif (appliances[rand_app[0]] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app = np.random.randint(len(appliances) - 3, size=1)

                    if (appliances[rand_app2] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app2 = 1
                    elif (appliances[rand_app2] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app2 = 0
                    elif (appliances[rand_app2] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app2 = 1
                    elif (appliances[rand_app2] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app2 = 0
                    elif (appliances[rand_app2] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app2 = 1

                    if (appliances[rand_app3] == 'kettle' and sample_count_k >= samples_per_class):
                        rand_app3 = 0
                    elif (appliances[rand_app3] == 'microwave' and sample_count_m >= samples_per_class):
                        rand_app3 = 1
                    elif (appliances[rand_app3] == 'fridge' and sample_count_f >= samples_per_class):
                        rand_app3 = 0
                    elif (appliances[rand_app3] == 'washing machine' and sample_count_w >= samples_per_class):
                        rand_app3 = 1
                    elif (appliances[rand_app3] == 'dish washer' and sample_count_d > samples_per_class):
                        rand_app3 = 0


                    rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
                    app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
                    app2 = app2.reset_index()
                    app2 = app2.drop(["index"], axis=1)
                    rand_shift = np.random.randint(number_shift, size=1)
                    app2_series = padd_shift(rand_shift[0], app2)
                    app2_series = app2_series.add(s_app, fill_value=0)

                    rand_3 = np.random.randint(len(activations_[appliances[rand_app2]]), size=1)
                    app3 = activations_[appliances[rand_app2]][rand_3[0]]
                    app3 = app3.reset_index()
                    app3 = app3.drop(["index"], axis=1)
                    rand_shift = np.random.randint(number_shift, size=1)
                    app3_series = padd_shift(rand_shift[0], app3)
                    app3_series = app3_series.add(s_app, fill_value=0)

                    rand_4 = np.random.randint(len(activations_[appliances[rand_app3]]), size=1)
                    app4 = activations_[appliances[rand_app3]][rand_4[0]]
                    app4 = app4.reset_index()
                    app4 = app4.drop(["index"], axis=1)
                    rand_shift = np.random.randint(number_shift, size=1)
                    app4_series = padd_shift(rand_shift[0], app4)
                    app4_series = app4_series.add(s_app, fill_value=0)

                    vector = app2_series.add(app1['power']['active'], fill_value=0)
                    vector = vector.add(app3_series, fill_value=0)
                    vector = vector.add(app4_series, fill_value=0)
                    aggregate = vector.add(s, fill_value=0)
                    aggregate = aggregate.to_numpy()
                    aggregate[aggregate<0] = 0
                    aggregate = pd.Series(aggregate)
                    aggregate_.append(aggregate)

                    sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(
                        appliances[i], appliances[rand_app[0]], appliances[rand_app2], appliances[rand_app3], 0, sample_count_k, sample_count_m,
                        sample_count_f, sample_count_w, sample_count_d, app1['power']['active'], app2['power']['active'],
                        app3['power']['active'], app4['power']['active'], 0)

                    #plt.plot(aggregate)
                    #plt.show()

                    list_vectors_strong = strong_labels_creation(appliance1=appliances[i], app1=app1_series,
                                                                 appliance2=appliances[rand_app[0]], app2=app2_series,
                                                                 appliance3=appliances[rand_app2], app3=app3_series,
                                                                 appliance4=appliances[rand_app3], app4=app4_series, appliance5=0,
                                                                 app5=0, flag=4)
                    final_strong.append(list_vectors_strong)
                    vector_weak = [0, 0, 0, 0, 0]
                    vector_weak[i] = 1
                    vector_weak[rand_app[0]] = 1
                    vector_weak[rand_app2] = 1
                    vector_weak[rand_app3] = 1
                    print(vector_weak)
                    final_weak.append(vector_weak)
                    indexes_annotation(appliances[i], appliances[rand_app[0]], appliances[rand_app2], appliances[rand_app3], 0, rand_1[0],
                                       rand_2[0], rand_3[0], rand_4[0], 0)

            else:
                break

    indices_of_activations['kettle'] = list_kettle
    indices_of_activations['microwave'] = list_micro
    indices_of_activations['fridge'] = list_fridge
    indices_of_activations['washing_machine'] = list_wash
    indices_of_activations['dish_washer'] = list_dish

    repetitions_k, repetitions_m, repetitions_f, repetitions_w, repetitions_d = repetition_counter(appliances)
    print("repetitions counted!")


    with open(destination_path + 'phase_repetition_' + str(building) +'_.txt', 'a+') as file:
         print("Repetitions Kettle:", file=file)
         print(repetitions_k, file=file)
         print("Repetitions Micro:", file=file)
         print(repetitions_m, file=file)
         print("Repetitions Fridge:", file=file)
         print(repetitions_f, file=file)
         print("Repetitions Wash:", file=file)
         print(repetitions_w, file=file)
         print("Repetitions Dish:", file=file)
         print(repetitions_d, file=file)

    return aggregate_, final_strong, final_weak

def aggregate_creation(appliances = None, building= None):

    global samples_per_class
    activations_ = activation_appliances_nilmtk(appliances, building)


    if building == 1:
        samples_per_class = samples_per_class_1
    if building == 2:
        samples_per_class = samples_per_class_2
    if building == 5:
        samples_per_class = samples_per_class_5
    if building == 4:
        samples_per_class = samples_per_class_4
    if building == 3:
        samples_per_class = samples_per_class_3

    aggregate, final_strong, final_weak = data_iteration(activations_, samples_per_class=samples_per_class, building=arguments.building)

    return aggregate, final_strong,final_weak

if __name__ == "__main__":
        window_length = arguments.window_length
        building = arguments.building
        print("Building:", building)
        aggregate, final_strong,final_weak = aggregate_creation(appliances = appliances,building = arguments.building)

        if building == 1:
            dict_ = dict_1
        if building == 2:
            dict_ = dict_2
        if building == 3:
            dict_ = dict_3
        if building == 4:
            dict_ = dict_4
        if building == 5:
            dict_ = dict_5


        for bag in range(len(aggregate)):
            agg = aggregate[bag].to_numpy()

            strong = final_strong[bag]

            # data correction for anomalous activations length

            if len(agg) > 2550 or len(strong[0]) > 2550 or len(strong[1]) > 2550 or len(strong[2]) > 2550 or len(
                    strong[3]) > 2550 or len(strong[4]) > 2550:
                continue

            else:
                np.save("../aggregate_data/house_" + str(building) + "/aggregate_%d" % bag, agg)



                for k in range(len(strong)):
                    strong[k] = strong[k].tolist()
                weak = final_weak[bag]
                label = 'labels_%d' % bag
                dict_[label] = {'strong': [], 'weak': []}
                dict_[label]['strong'] = strong
                dict_[label]['weak'] = weak

        with open('../labels_'+ str(building) +'.json', 'w') as outfile:
                json.dump(dict_, outfile)

        print("Total number of bags:",len(final_strong))
        del final_weak
        del final_strong
        del aggregate

        gc.collect()



