import numpy as np
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt
from nilmtk.nilmtk.dataset import DataSet
from nilmtk.nilmtk.electric import Electric
from nilmtk.nilmtk.dataset_converters.refit import convert_refit
from refit_appliance_info import *

on_power_threshold = {
    'washing machine': 25,
    'washing_machine':25,
    'dish_washer':25,
    'dish washer': 25,
    'microwave': 100,
    'kettle': 2000,
    'fridge': 80
}

min_off_duration = {
    'washing machine': 300,
    'dish washer': 1800,
    'microwave': 20,
    'kettle': 20,
    'fridge': 20
}

min_on_duration = {
    'washing machine': 600,
    'dish washer': 600,
    'microwave': 20,
    'kettle': 20,
    'fridge': 20
}
indices_of_activations = {
    'washing_machine': [],
    'kettle': [],
    'fridge': [],
    'dish_washer': [],
    'microwave': []
}

app_dict = {'kettle': {'house': [2,3,4,5,6,7,8,9,12,13,19],
                    'channel':  [8,9,9,8,7,9,9,7,6,9,9]},
              'microwave': {'house': [4,10,12,17,19],
                        'channel': [8,8,5,9,8]},
              'fridge': {'house':  [2,5,9,12,15],
                         'channel': [1,1,1,1,1]},
              'washing machine': {'house':  [2,5,7,8,9, 13, 15, 16,    17,    18],
                                  'channel':[2,3,5,4,3, 3,  5,   4,    5,     2]},
              'dish washer': {'house':    [2,5,7,9,13],
                              'channel':  [3,4,6,4,4]}
              }

np.random.seed(0)

#refit_ = ''
refit_path = "../REFIT.h5"
#convert_refit.convert_refit(refit_,refit_path, 'HDF' )
refit = DataSet(refit_path)
houses = [2,3,4,5,6,7,8,9,10,12,13,15,16,17,18,19]

number_shift = 50
number_shift1 = 600
number_shift2 = 1000
number_shift4 = 1750
window_lenght = 2550
phase_change = 5
window_length = 2550
list_kettle = []
list_micro = []
list_fridge = []
list_wash = []
list_dish = []

def indexes_annotation(app1,app2,app3,app4,app5, rand_1, rand_2,rand_3, rand_4, rand_5):


    if app1 == 'kettle':
        list_kettle.append(rand_1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'kettle':
        list_kettle.append(rand_2)
        #indices_of_activations[app2] = rand_2
    elif app3 == 'kettle':
        list_kettle.append(rand_3)
        #indices_of_activations[app3] = rand_3
    elif app4 == 'kettle':
        list_kettle.append(rand_4)
        #indices_of_activations[app4] = rand_4
    elif app5 == 'kettle':
        list_kettle.append(rand_5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'microwave':
        list_micro.append(rand_1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'microwave':
        list_micro.append(rand_2)
        #indices_of_activations[app2] = rand_2
    elif app3 == 'microwave':
        list_micro.append(rand_3)
        #indices_of_activations[app3] = rand_3
    elif app4 == 'microwave':
        list_micro.append(rand_4)
        #indices_of_activations[app4] = rand_4
    elif app5 == 'microwave':
        list_micro.append(rand_5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'fridge':
        list_fridge.append(rand_1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'fridge':
        list_fridge.append(rand_2)
        #indices_of_activations[app2] = rand_2
    elif app3 == 'fridge':
        list_fridge.append(rand_3)
        #indices_of_activations[app3] = rand_3
    elif app4 == 'fridge':
        list_fridge.append(rand_4)
        #indices_of_activations[app4] = rand_4
    elif app5 == 'fridge':
        list_fridge.append(rand_5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'washing_machine':
        list_wash.append(rand_1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'washing_machine':
        list_wash.append(rand_2)
        #indices_of_activations[app2] = rand_2
    elif app3 == 'washing_machine':
        list_wash.append(rand_3)
        #indices_of_activations[app3] = rand_3
    elif app4 == 'washing_machine':
        list_wash.append(rand_4)
        #indices_of_activations[app4] = rand_4
    elif app5 == 'washing_machine':
        list_wash.append(rand_5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'dish_washer':
        list_dish.append(rand_1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'dish_washer':
        list_dish.append(rand_2)
        #indices_of_activations[app2] = rand_2
    elif app3 == 'dish_washer':
        list_dish.append(rand_3)
        #indices_of_activations[app3] = rand_3
    elif app4 == 'dish_washer':
        list_dish.append(rand_4)
        #indices_of_activations[app4] = rand_4
    elif app5 == 'dish_washer':
        list_dish.append(rand_5)
        #indices_of_activations[app5] = rand_5


def sample_counter(app1,app2,app3,app4,app5,sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d, act1,act2,act3,act4,act5):


    if app1 == 'kettle':
        sample_count_k += len(act1)
        #indices_of_activations[app1] = rand_1
    if app2 == 'kettle':
        sample_count_k += len(act2)
        #indices_of_activations[app2] = rand_2
    if app3 == 'kettle':
        sample_count_k += len(act3)
        #indices_of_activations[app3] = rand_3
    if app4 == 'kettle':
        sample_count_k += len(act4)
        #indices_of_activations[app4] = rand_4
    if app5 == 'kettle':
        sample_count_k += len(act5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'microwave':
        sample_count_m += len(act1)
        #indices_of_activations[app1] = rand_1
    if app2 == 'microwave':
        sample_count_m += len(act2)
        #indices_of_activations[app2] = rand_2
    if app3 == 'microwave':
        sample_count_m += len(act3)
        #indices_of_activations[app3] = rand_3
    if app4 == 'microwave':
        sample_count_m += len(act4)
        #indices_of_activations[app4] = rand_4
    if app5 == 'microwave':
        sample_count_m += len(act5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'fridge':
        sample_count_f += len(act1)
        #indices_of_activations[app1] = rand_1
    if app2 == 'fridge':
        sample_count_f += len(act2)
        #indices_of_activations[app2] = rand_2
    if app3 == 'fridge':
        sample_count_f += len(act3)
        #indices_of_activations[app3] = rand_3
    if app4 == 'fridge':
        sample_count_f += len(act4)
        #indices_of_activations[app4] = rand_4
    if app5 == 'fridge':
        sample_count_f += len(act5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'washing_machine':
        sample_count_w += len(act1)
        #indices_of_activations[app1] = rand_1
    if app2 == 'washing_machine':
        sample_count_w += len(act2)
        #indices_of_activations[app2] = rand_2
    if app3 == 'washing_machine':
        sample_count_w += len(act3)
        #indices_of_activations[app3] = rand_3
    if app4 == 'washing_machine':
        sample_count_w += len(act4)
        #indices_of_activations[app4] = rand_4
    if app5 == 'washing_machine':
        sample_count_w += len(act5)
        #indices_of_activations[app5] = rand_5

    if app1 == 'dish_washer':
        sample_count_d += len(act1)
        #indices_of_activations[app1] = rand_1
    elif app2 == 'dish_washer':
        sample_count_d += len(act2)
        #indices_of_activations[app2] = rand_2
    if app3 == 'dish_washer':
        sample_count_d += len(act3)
        #indices_of_activations[app3] = rand_3
    if app4 == 'dish_washer':
        sample_count_d += len(act4)
        #indices_of_activations[app4] = rand_4
    if app5 == 'dish_washer':
        sample_count_d += len(act5)
        #indices_of_activations[app5] = rand_5

    return sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d


def repetition_counter(appliances):

    indexes_k = []
    indexes_m = []
    indexes_f = []
    indexes_w = []
    indexes_d = []
    list_indexes = [indexes_k, indexes_m, indexes_f, indexes_w, indexes_d]
    repetition = 0
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
                vector_strong_1_a = pd.Series(np.zeros(window_lenght))

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
                vector_strong_1_b = pd.Series(np.zeros(window_lenght))

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
                vector_strong_1_c = pd.Series(np.zeros(window_lenght))

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
                vector_strong_1_d = pd.Series(np.zeros(window_lenght))

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
                vector_strong_1_e = pd.Series(np.zeros(window_lenght))


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
                vector_strong_2_a = pd.Series(np.zeros(window_lenght))

            list_2 = []
            if appliance2 == 'microwave':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['microwave']:
                        list_2.append(0)
                    else:
                        list_2.append(1)
                vector_strong_2_b = pd.Series(list_2)
            else:
                vector_strong_2_b = pd.Series(np.zeros(window_lenght))

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
                vector_strong_2_c = pd.Series(np.zeros(window_lenght))

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
                vector_strong_2_d = pd.Series(np.zeros(window_lenght))

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
                vector_strong_2_e = pd.Series(np.zeros(window_lenght))

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
                vector_strong_3_a = pd.Series(np.zeros(window_lenght))

            list_3 = []
            if appliance2 == 'fridge':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['fridge']:
                        list_3.append(0)
                    else:
                        list_3.append(1)
                vector_strong_3_b = pd.Series(list_3)
            else:
                vector_strong_3_b = pd.Series(np.zeros(window_lenght))

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
                vector_strong_3_c = pd.Series(np.zeros(window_lenght))

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
                vector_strong_3_d = pd.Series(np.zeros(window_lenght))

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
                vector_strong_3_e = pd.Series(np.zeros(window_lenght))

            vector_strong_3 = vector_strong_3_a.add(vector_strong_3_b)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_c)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_d)
            vector_strong_3 = vector_strong_3.add(vector_strong_3_e)




            list_4 = []
            vector_strong_4 = pd.Series()
            if appliance1 == 'washing_machine':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['washing_machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)
                vector_strong_4 = pd.Series(list_4)
            elif appliance2 == 'washing_machine':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['washing_machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)
                vector_strong_4 = pd.Series(list_4)


            elif appliance3 == 'washing_machine':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['washing_machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 3 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)

            elif appliance4 == 'washing_machine':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['washing_machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 4 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)
            elif appliance5 == 'washing_machine':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['washing_machine']:
                        list_4.append(0)
                    else:
                        list_4.append(1)

                print("Appliance 5 vector strong washing machine")
                vector_strong_4 = pd.Series(list_4)
            else:
                vector_strong_4 = pd.Series(np.zeros(window_lenght))

            list_5 = []
            vector_strong_5 = pd.Series()
            if appliance1 == 'dish_washer':
                for l in range(len(app1)):
                    if app1[l] < on_power_threshold['dish_washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)
                vector_strong_5 = pd.Series(list_5)

            elif appliance2 == 'dish_washer':
                for l in range(len(app2)):
                    if app2[l] < on_power_threshold['dish_washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)
                vector_strong_5 = pd.Series(list_5)

            elif appliance3 == 'dish_washer':
                for l in range(len(app3)):
                    if app3[l] < on_power_threshold['dish_washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 3 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            elif appliance4 == 'dish_washer':
                for l in range(len(app4)):
                    if app4[l] < on_power_threshold['dish_washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 4 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            elif appliance5 == 'dish_washer':
                for l in range(len(app5)):
                    if app5[l] < on_power_threshold['dish_washer']:
                        list_5.append(0)
                    else:
                        list_5.append(1)

                print("Appliance 5 vector strong dish washer")
                vector_strong_5 = pd.Series(list_5)
            else:
                vector_strong_5 = pd.Series(np.zeros(window_lenght))




    list_vectors_strong.append(vector_strong_1)
    list_vectors_strong.append(vector_strong_2)
    list_vectors_strong.append(vector_strong_3)
    list_vectors_strong.append(vector_strong_4)
    list_vectors_strong.append(vector_strong_5)

    return list_vectors_strong


def activation_appliances_nilmtk(appliances, buildings):
    for appliance in appliances:
        for i in range(len(app_dict[appliance]['house'])):

            elec = refit.buildings[app_dict[appliance]['house'][i]].elec
            #lista = elec.appliances
            #app = lista[1]
            #curr_appliance = elec[appliance] elec.meters[8]

            activation_ = Electric.get_activations(elec.meters[app_dict[appliance]['channel'][i]], min_off_duration=min_off_duration[appliance],min_on_duration=min_on_duration[appliance],on_power_threshold=on_power_threshold[appliance])


            plt.plot(activation_[0])
            # plt.title(appliance + str(app_dict[appliance]['house'][i]))
            plt.show()
            # print("len activation")
            # print(len(activation_[0]))

            # df = next(elec.meters[app_dict[appliance]['channel'][i]].load(ac_type='active')) #, sample_period=8))
            # df.head()
            print("Casa:", app_dict[appliance]['house'][i])
            print(appliance)
            print("N° di attivazioni:",len(activation_))
            if appliance == 'dish washer':   #app_dict[appliance]['house'][i]
                np.save("/mnt/sda1/home/gtanoni/codice_prova/dish_washeractivations_" + str(app_dict[appliance]['house'][i]) + ".npy", activation_)
            elif appliance == 'washing machine':
                np.save("/mnt/sda1/home/gtanoni/codice_prova/washing_machineactivations_" + str(app_dict[appliance]['house'][i]) + ".npy", activation_)
            else:
                np.save("/mnt/sda1/home/gtanoni/codice_prova/" + appliance + "activations_" + str(app_dict[appliance]['house'][i]) + ".npy", activation_)


def data_iteration_seq3(activations_, build, appliances, num_of_bags):

    final_weak = []
    final_strong = []
    aggregate_ = []
    sample_count_k = 0
    sample_count_m = 0
    sample_count_f = 0
    sample_count_w = 0
    sample_count_d = 0


    for a in range(num_of_bags):

            indx = []
            for app in range(len(appliances)):
                if appliances[app] == '':
                    continue
                else:
                    indx.append(app)

            mu, sigma = 0, 1
            s = np.random.normal(mu, sigma, window_lenght)
            s = pd.Series(s)
            s_app = pd.Series(np.zeros(window_lenght))

            i = indx[0]
            print("Appliance 1")
            print(appliances[i])


            rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
            app1 = activations_[appliances[i]][rand_1[0]]
            app1 = app1.reset_index()
            app1 = app1.drop(["Unix"], axis=1)
            #app1_series = app1['power']['active']
            rand_shift = np.random.randint(number_shift, size=1)
            app1_series = padd_shift(rand_shift[0], app1)
            app1_series = app1_series.add(s_app, fill_value=0)


            rand_app2 = indx[1]


            rand_app = [indx[2]]
            print("Appliance 2")
            print(appliances[rand_app[0]])

            print("Appliance 3")
            print(appliances[rand_app2])

            rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
            app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
            app2 = app2.reset_index()
            app2 = app2.drop(["Unix"], axis=1)

            rand_shift = np.random.randint(number_shift1,number_shift2, size=1)
            app2_series = padd_shift(rand_shift[0], app2)
            app2_series = app2_series.add(s_app, fill_value=0)

            rand_3 = np.random.randint(len(activations_[appliances[rand_app2]]), size=1)
            app3 = activations_[appliances[rand_app2]][rand_3[0]]
            app3 = app3.reset_index()
            app3 = app3.drop(["Unix"], axis=1)

            sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d = sample_counter(appliances[i], appliances[rand_app[0]], appliances[rand_app2], 0, 0, sample_count_k, sample_count_m, sample_count_f, sample_count_w, sample_count_d, app1['power']['active'], app2['power']['active'], app3['power']['active'],0,0)

            rand_shift = np.random.randint(1200,number_shift4, size=1)
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


            print("Sequences with 3 appliances:")
            print(a)

    indices_of_activations['kettle'] = list_kettle
    indices_of_activations['microwave'] = list_micro
    indices_of_activations['fridge'] = list_fridge
    indices_of_activations['washing_machine'] = list_wash
    indices_of_activations['dish_washer'] = list_dish

    repetitions_k, repetitions_m, repetitions_f, repetitions_w, repetitions_d = repetition_counter(appliances)


    print("Total samples kettle:", sample_count_k)
    print("Total samples micro:", sample_count_m)
    print("Total samples fridge:", sample_count_f)
    print("Total samples washing:", sample_count_w)
    print("Total samples dish:", sample_count_d)
    #
    with open('/mnt/sda1/home/gtanoni/codice_prova/phase_repetition_' + str(build) +'_.txt', 'a+') as file:
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

def data_iteration_seq4(activations_, build, appliances, num_of_bags):
    final_weak = []
    final_strong = []
    aggregate_ = []
    sample_count_k = 0
    sample_count_m = 0
    sample_count_f = 0
    sample_count_w = 0
    sample_count_d = 0

    for a in range(num_of_bags):


        mu, sigma = 0, 1
        s = np.random.normal(mu, sigma, window_lenght)
        s = pd.Series(s)
        s_app = pd.Series(np.zeros(window_lenght))

        indx = []
        for app in range(len(appliances)):
            if appliances[app] == '':
                continue
            else:
                indx.append(app)

        i = indx[0]
        print("Appliance 1")
        print(appliances[i])
        rand_app = [indx[2]]


        rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
        app1 = activations_[appliances[i]][rand_1[0]]
        app1 = app1.reset_index()
        app1 = app1.drop(["Unix"], axis=1)
        app1_series = app1['power']['active']
        app1_series = app1_series.add(s_app, fill_value=0)
        rand_app2 = indx[1]
        print(appliances[rand_app2])

        print("Appliance 2")
        print(appliances[rand_app[0]])
        print("Appliance 3")
        rand_app3 = indx[3]
        print("Appliance 4")
        print(appliances[rand_app3])

        rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
        app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
        app2 = app2.reset_index()
        app2 = app2.drop(["Unix"], axis=1)
        rand_shift = np.random.randint(0,number_shift, size=1)
        app2_series = padd_shift(rand_shift[0], app2)
        app2_series = app2_series.add(s_app, fill_value=0)

        rand_3 = np.random.randint(len(activations_[appliances[rand_app2]]), size=1)
        app3 = activations_[appliances[rand_app2]][rand_3[0]]
        app3 = app3.reset_index()
        app3 = app3.drop(["Unix"], axis=1)
        rand_shift = np.random.randint(number_shift1,number_shift2, size=1)
        app3_series = padd_shift(rand_shift[0], app3)
        app3_series = app3_series.add(s_app, fill_value=0)

        rand_4 = np.random.randint(len(activations_[appliances[rand_app3]]), size=1)
        app4 = activations_[appliances[rand_app3]][rand_4[0]]
        app4 = app4.reset_index()
        app4 = app4.drop(["Unix"], axis=1)
        rand_shift = np.random.randint(number_shift4,2000, size=1)
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

    indices_of_activations['kettle'] = list_kettle
    indices_of_activations['microwave'] = list_micro
    indices_of_activations['fridge'] = list_fridge
    indices_of_activations['washing_machine'] = list_wash
    indices_of_activations['dish_washer'] = list_dish

    repetitions_k, repetitions_m, repetitions_f, repetitions_w, repetitions_d = repetition_counter(appliances)
    print("repetitions counted!")
    print("Total samples kettle:", sample_count_k)
    print("Total samples micro:", sample_count_m)
    print("Total samples fridge:", sample_count_f)
    print("Total samples washing:", sample_count_w)
    print("Total samples dish:", sample_count_d)
    #
    with open('/mnt/sda1/home/gtanoni/codice_prova/phase_repetition_' + str(build) + '_.txt',
              'a+') as file:
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

def data_iteration_seq2(activations_, build, appliances, num_of_bags):

    final_weak = []
    final_strong = []
    aggregate_ = []
    sample_count_k = 0
    sample_count_m = 0
    sample_count_f = 0
    sample_count_w = 0
    sample_count_d = 0

    for a in range(num_of_bags):

        indx = []
        for app in range(len(appliances)):
            if appliances[app] == '':
                continue
            else:
                indx.append(app)
        list_indices_app1 = []
        list_indices_app2 = []


        mu, sigma = 0, 1
        s = np.random.normal(mu, sigma,window_lenght)
        s = pd.Series(s)
        s_app = pd.Series(np.zeros(window_lenght))

        i = indx[0]
        print("Appliance 1")
        print(appliances[i])

        rand_app = [indx[1]]

        print("Appliance 2")
        print(appliances[rand_app[0]])

        rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)
        list_indices_app1.append(rand_1)
        app1 = activations_[appliances[i]][rand_1[0]]
        app1 = app1.reset_index()
        app1 = app1.drop(["Unix"], axis=1)
        app1_series = app1['power']['active']

        rand_2 = np.random.randint(len(activations_[appliances[rand_app[0]]]), size=1)
        list_indices_app2.append(rand_2)
        app2 = activations_[appliances[rand_app[0]]][rand_2[0]]
        app2 = app2.reset_index()
        app2 = app2.drop(["Unix"], axis=1)

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

            #plt.plot(aggregate)
            #plt.show()

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



    indices_of_activations['kettle'] = list_kettle
    indices_of_activations['microwave'] = list_micro
    indices_of_activations['fridge'] = list_fridge
    indices_of_activations['washing_machine'] = list_wash
    indices_of_activations['dish_washer'] = list_dish

    repetitions_k, repetitions_m, repetitions_f, repetitions_w, repetitions_d = repetition_counter(appliances)
    print("repetitions counted!")


    print("Total samples kettle:", sample_count_k)
    print("Total samples micro:", sample_count_m)
    print("Total samples fridge:", sample_count_f)
    print("Total samples washing:", sample_count_w)
    print("Total samples dish:", sample_count_d)
    #
    with open('/mnt/sda1/home/gtanoni/codice_prova/phase_repetition_' + str(build) + '_.txt',
              'a+') as file:
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


def data_iteration_seq1(activations_, build, appliances, num_of_bags):
    sample_count_k = 0
    sample_count_m = 0
    sample_count_f = 0
    sample_count_w = 0
    sample_count_d = 0
    final_weak = []
    final_strong = []
    aggregate_ = []

    for a in range(num_of_bags):  # l'AGGREGATO è LA SOMMA DI 1 APPLIANCE

            mu, sigma = 0, 1
            s = np.random.normal(mu, sigma, window_lenght)
            s = pd.Series(s)
            s_app = pd.Series(np.zeros(window_lenght))

            indx = []
            for app in range(len(appliances)):
                if appliances[app] == '':
                    continue
                else:
                    indx.append(app)

            i = indx[0]

            print("Appliance 1")
            print(appliances[i])

            rand_1 = np.random.randint(len(activations_[appliances[i]]), size=1)

            app1 = activations_[appliances[i]][rand_1[0]]
            app1 = app1.reset_index()
            app1 = app1.drop(["Unix"], axis=1)
            app1_series = app1['power']['active']

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
            indexes_annotation(appliances[i], 0, 0, 0, 0, rand_1[0], 0, 0, 0, 0)


    indices_of_activations['kettle'] = list_kettle
    indices_of_activations['microwave'] = list_micro
    indices_of_activations['fridge'] = list_fridge
    indices_of_activations['washing_machine'] = list_wash
    indices_of_activations['dish_washer'] = list_dish

    repetitions_k, repetitions_m, repetitions_f, repetitions_w, repetitions_d = repetition_counter(appliances)
    print("repetitions counted!")

    print("Total samples kettle:", sample_count_k)
    print("Total samples micro:", sample_count_m)
    print("Total samples fridge:", sample_count_f)
    print("Total samples washing:", sample_count_w)
    print("Total samples dish:", sample_count_d)
    #
    with open('../phase_repetition_' + str(build) + '_.txt',
              'a+') as file:
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

appliances = ['kettle', 'microwave','fridge', 'washing machine', 'dish washer']
# Save activations' files
activation_appliances_nilmtk(appliances,houses)

dict_ = {'2': ['kettle', '', 'fridge', 'washing_machine', 'dish_washer'],
         '3': ['kettle', '', '','',''],
         '4': ['kettle', 'microwave', '','',''],
         '5': ['kettle','', 'fridge', 'washing_machine', 'dish_washer'],
         '6': ['kettle', '', '','',''],
         '7': ['kettle','', '', 'washing_machine', 'dish_washer'],
         '8': ['kettle','', '', 'washing_machine', ''],
         '9': ['kettle','', '', 'washing_machine', 'dish_washer'],
         '10':['', 'microwave', '','',''],
         '12': ['', 'microwave', 'fridge', '',''],
         '13': ['kettle', '', '', '', 'dish_washer'],
         '15': ['','', 'fridge', '', ''],
         '16': ['', '', '', 'washing_machine', 'dish_washer'],
         '17': ['', 'microwave', '', 'washing_machine', ''],
         '18': ['','','','washing_machine', 'dish_washer'],
         '19': ['kettle', 'microwave'],
         '20': ['kettle','','','', 'dish_washer']}

for id in houses:
    list_activations = {'kettle': [], 'microwave': [], 'fridge': [], 'washing_machine': [], 'dish_washer': []}

    for app in dict_[str(id)]:

        if app == '':
            continue
        else:
            list_new = []

            activations = np.load('../'+ app +'activations_' + str(id) + '.npy', allow_pickle=True)
            for k in range(len(activations)):
                new = activations[k].resample('8S').bfill()
                list_new.append(new)
            np.save('../'+ app +'activations_' + str(id) + '_resampled.npy', list_new)
            list_activations[app] = list_new
    if id == 2 or id == 5 or id == 7:
        Aggregate_, final_strong_, final_weak_ = data_iteration_seq4(list_activations, id, dict_[str(id)], num_of_bags=8000)
    if id == 3 or id==6 or id==10 or id==16 or id==18 or id==15:
        Aggregate_, final_strong_, final_weak_ = data_iteration_seq1(list_activations,  id, dict_[str(id)], num_of_bags=12000)
    if id == 4 or id == 8 or id == 12 or id == 17 or id == 19:
        Aggregate_, final_strong_, final_weak_ = data_iteration_seq2(list_activations, id, dict_[str(id)], num_of_bags=12000)
    if id == 9:
        Aggregate_, final_strong_, final_weak_ = data_iteration_seq3(list_activations,  id, dict_[str(id)], num_of_bags=9000)

    #    SAVE THE DATA AND LABELS   #
    for bag in range(len(Aggregate_)):
        agg = Aggregate_[bag].to_numpy()


        strong = final_strong_[bag]

        if len(agg) > 2550 or len(strong[0]) != 2550 or len(strong[1]) != 2550 or len(strong[2]) !=2550 or len(
                strong[3]) != 2550 or len(strong[4]) != 2550:
            continue

        else:
            np.save("../aggregate_data/house_"+ str(id) +"/aggregate_%d" % bag,agg)
            strong = final_strong_[bag]
            for k in range(len(strong)):
                strong[k] = strong[k].tolist()
            weak = final_weak_[bag]
            np.save("../labels/house_"+ str(id) +"/strong_labels_%d.npy" % bag,strong)
            np.save("../labels/house_"+ str(id) +"/weak_labels_%d.npy" % bag,weak)
    print("Total number of bags:",len(final_weak_))
    del final_strong_
    del final_weak_
    del Aggregate_






