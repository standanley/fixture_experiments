fname = ''
import pickle
f = open(fname, 'r')
debug_dict = pickle.load(f)


import matplotlib.pyplot as plt
leg = []
bump = 0
for p, r in debug_dict.items():
    leg.append(p)
    plt.plot(r.value[0], r.value[1] + bump, '-+')
    bump += 0.0 # useful for separating clock signals
plt.grid()
plt.legend(leg)
plt.show()
