import argparse
from matplotlib import pyplot as plt
from nilmtk.nilmtk.dataset import DataSet
from utils_dataset_test import *


parser = argparse.ArgumentParser(description="Noise extraction")
parser.add_argument("--refit", type=bool, default=True, help="REFIT dataset noise extraction")
parser.add_argument("--ukdale", type=bool, default=True, help="UKDALE dataset noise extraction")
parser.add_argument("--building", type=int, default=1, help="UKDALE dataset noise extraction")
parser.add_argument("--start", type=str, default='2013-09-17', help="Start date")
parser.add_argument("--end", type=str, default='2015-07-08', help="End date")
arguments = parser.parse_args()

if __name__ == "__main__":
    print(arguments.building)
    refit_ = arguments.refit
    if refit_:
        print("REFIT Noise extraction")
        app_dict_refit = {'kettle': {2: 8, 3:9, 4:9,5:8, 6:7, 7:9, 8:9,9:7, 12:6,13:9,16:8,18:5,19:9},
                      'microwave': {2:5, 3:8, 4:8, 5:8, 6:6,8:8,9:6, 10:7, 12:5, 13:7, 16:7, 17:9,18:4,19:8},
                      'fridge': {2:1,3:2,4:1,5:1, 7:1,8:1, 9:1,10:4, 12:1, 15:1, 16:2, 17:1, 18:1, 19:1},
                      'washing machine': {2:2,3:6,4:5,5:3,6:2, 7:5,8:4,9:3,10:5, 13:3, 15:5,  16:4,    17:5,    18:2,  19:4},
                      'dish washer': {2:3,3:5, 5:4,6:3,  7:6, 9:4, 10:6, 13:4, 15:6, 17:6, 19:5}}

        appliances = {2: ['kettle', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     3: ['kettle', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     4: ['kettle', 'microwave', 'fridge', 'washing_machine',''],
                     5: ['kettle', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     6: ['kettle', 'microwave', '', 'washing_machine', 'dish_washer'],
                     7: ['kettle','', 'fridge', 'washing_machine', 'dish_washer'],
                     8: ['kettle','microwave', 'fridge', 'washing_machine', ''],
                     9: ['kettle', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     10: ['', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     12: ['kettle', 'microwave', 'fridge', '',''],
                     13: ['kettle', '', 'fridge', 'washing_machine', 'dish_washer'],
                     15: ['','', 'fridge', 'washing_machine', 'dish_washer'],
                     16: ['kettle', 'microwave', 'fridge', 'washing_machine', ''],
                     17: ['', 'microwave', 'fridge', 'washing_machine', 'dish_washer'],
                     18: ['kettle','microwave','fridge','washing_machine', ''],
                     19: ['kettle', 'microwave', 'fridge', 'washing_machine', 'dish_washer']}

        refit_path = "../REFIT.h5"
        build = arguments.building
        refit = DataSet(refit_path)
        # insert desired period
        refit.set_window(start=arguments.start, end=arguments.end)
        elec = refit.buildings[build].elec
        good = elec.good_sections(full_results=False)
        period = 8
        noised = []

        for i in range(len(good)):
            try:
                refit.set_window(good[i])
                elec = refit.buildings[build].elec


                mains_ = next(elec.mains().load(sample_period=period))
                mains_ = mains_['power']['active'].to_numpy()
                if appliances[build][0] == 'kettle':
                    kettle = next(elec.meters[app_dict_refit['kettle'][build]].load(sample_period=period))
                    kettle = kettle['power']['active'].to_numpy()
                else:
                    kettle =  np.zeros(len(mains_))
                if appliances[build][1] == 'microwave':
                    micro = next(elec.meters[app_dict_refit['microwave'][build]].load(sample_period=period))
                    micro = micro['power']['active'].to_numpy()
                else:
                    micro = np.zeros(len(mains_))
                if appliances[build][3] == 'fridge':
                    fridge = next(elec.meters[app_dict_refit['fridge'][build]].load(sample_period=period))
                    fridge = fridge['power']['active'].to_numpy()
                else:
                    fridge =  np.zeros(len(mains_))
                if appliances[build][3] == 'washing_machine':
                    wash = next(elec.meters[app_dict_refit['washing machine'][build]].load(sample_period=period))
                    wash = wash['power']['active'].to_numpy()
                else:
                    wash =  np.zeros(len(mains_))
                if appliances[build][4] == 'dish_washer':
                    dish = next(elec.meters[app_dict_refit['dish washer'][build]].load(sample_period=period))
                    dish = dish['power']['active'].to_numpy()
                else:
                    dish =  np.zeros(len(mains_))

                plt.plot(mains_)
                plt.plot(kettle)
                plt.plot(micro)
                plt.plot(fridge)
                plt.plot(wash)
                plt.plot(dish)
                plt.show()

                # noise creation
                sum = np.zeros(len(mains_))
                sum = np.add(kettle,sum)
                sum = np.add(micro, sum)
                sum = np.add(fridge, sum)
                sum = np.add(wash, sum)
                sum = np.add(dish, sum)
                noise = mains_ - sum

                # negative values cancellation or misalignment correction
                for s in range(len(noise) - 30):
                    if (noise[s] < 0 and noise[s + 1] > 0):
                        for po in range(s,s + 30):
                            noise[po] = 1
                for s in range(len(noise) - 30):
                    if (noise[s] >= 0 and noise[s + 1] < 0):
                        for po in range(s - 30, s):
                            noise[po] = 1

                noised.append(noise)

            except StopIteration:
                continue
        print("Done!")
        np.save('../noise_' + str(build) + '.npy', noised)
    ukdale = arguments.ukdale
    if ukdale:

        print("UKDALE Noise extraction")
        ukdale_path = "../ukdale.h5"
        build = arguments.building
        ukdale = DataSet(ukdale_path)
        # insert desired period
        ukdale.set_window(start=arguments.start, end=arguments.end)
        elec = ukdale.buildings[build].elec
        good = elec.good_sections(full_results=False)
        period = 6
        noised = []

        for i in range(len(good)):
            try:
                ukdale.set_window(good[i])
                elec = ukdale.buildings[build].elec

                mains_ = next(elec.mains().load(sample_period=period))

                if mains_.shape[0] == 0 and mains_.shape[1] == 0:
                    continue
                else:
                    Mains_ = mains_['power']['active'].to_numpy()
                if build != 3 or build != 4:
                    kettle = next(elec['kettle'].load(sample_period=period))
                    kettle = kettle['power']['active'].to_numpy()


                    micro = next(elec['microwave'].load(sample_period=period))
                    micro = micro['power']['active'].to_numpy()


                    fridge = next(elec['fridge'].load(sample_period=period))
                    fridge = fridge['power']['active'].to_numpy()



                    wash = next(elec['washing machine'].load(sample_period=period))
                    wash = wash['power']['active'].to_numpy()


                    dish = next(elec['dish washer'].load(sample_period=period))
                    dish = dish['power']['active'].to_numpy()

                else:
                    if build == 4:
                        fridge = next(elec['fridge'].load(sample_period=period))
                        fridge = fridge['power']['active'].to_numpy()
                        dish = np.zeros(len(mains_))
                        kettle = np.zeros(len(mains_))
                        wash = np.zeros(len(mains_))
                        micro = np.zeros(len(mains_))
                    else:
                        kettle = next(elec['kettle'].load(sample_period=period))
                        kettle = kettle['power']['active'].to_numpy()
                        dish = np.zeros(len(mains_))
                        fridge = np.zeros(len(mains_))
                        wash = np.zeros(len(mains_))
                        micro = np.zeros(len(mains_))
                plt.plot(Mains_)
                plt.plot(kettle)
                plt.plot(micro)
                plt.plot(fridge)
                plt.plot(wash)
                plt.plot(dish)
                plt.show()

                # noise creation
                if len(kettle) == len(micro) == len(fridge) == len(wash) == len(dish) == len(Mains_):
                    sum = np.zeros(len(Mains_))
                    sum = np.add(kettle, sum)
                    sum = np.add(micro, sum)
                    sum = np.add(fridge, sum)
                    sum = np.add(wash, sum)
                    sum = np.add(dish, sum)
                    noise = Mains_ - sum

                    # negative values cancellation or misalignment correction
                    for s in range(len(noise) - 30):
                        if (noise[s] < 0 and noise[s + 1] > 0):
                            for po in range(s, s + 30):
                                noise[po] = 1
                    for s in range(len(noise) - 30):
                        if (noise[s] >= 0 and noise[s + 1] < 0):
                            for po in range(s - 30, s):
                                noise[po] = 1

                    noised.append(noise)
                else:
                    continue

            except StopIteration:
                continue
        print("Done!")
        np.save('../noise_' + str(build) + '.npy', noised)
