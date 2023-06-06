from run import *
import json, os


# pairs (q, L) to consider
qs_Ls = [(2, 10), (2, 20), (2, 30), (4, 20), (5, 20), (8, 10), (8, 20), (8, 30)]

# results path
path = "results"


# round up temperatures
# for (q, L) in qs_Ls:
#
#     with open(os.path.join(path, f"simulation_{q=}_{L=}.json")) as f:
#
#         d = json.load(f)
#         spec_heat_temps = d['spec_heat_temps']
#
#         for i, x in enumerate(spec_heat_temps):
#             spec_heat_temps[i] = round(x, 3)
#
#     with open(os.path.join(path, f"simulation_{q=}_{L=}.json"), 'w') as f:
#         json.dump(d, f)


# for (q, L) in qs_Ls:
#
#     with open(os.path.join(path, f"simulation_{q=}_{L=}.json")) as f:
#
#         # dict_keys(['temps', 'avg_en', 'avg_mag', 'spec_heat', 'spec_heat_temps'])
#         d = json.load(f)
#
#         print(d['temps'])
#         print(d['avg_en'])
#         print(d['avg_mag'])
#         print(d['spec_heat'])
#
#         for i in [1, 0]:
#
#             t = d['temps'][i]
#             nstep = 2*10**7
#             burnin = 2*10**7
#
#             en, mag = MCMC(L, q, t, nstep, burnin)
#
#             d['avg_en'][i] = en
#             d['avg_mag'][i] = mag
#             d['spec_heat'][i] = (d['avg_en'][i + 1] - d['avg_en'][i]) / (d['temps'][i + 1] - d['temps'][i])
#
#     with open(os.path.join(path, f"simulation_{q=}_{L=}.json"), 'w') as f:
#         json.dump(d, f)


for (q, L) in qs_Ls:

    with open(os.path.join(path, f"simulation_{q=}_{L=}.json")) as f:

        # dict_keys(['temps', 'avg_en', 'avg_mag', 'spec_heat', 'spec_heat_temps'])
        d = json.load(f)

        print(d['avg_mag'])
