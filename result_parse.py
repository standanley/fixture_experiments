import csv
import numpy as np
import fault
import scipy.interpolate
import matplotlib.pyplot as plt

filename = './selected_v2t_results_vcsv.vcsv'

def read_vcsv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        reader = list(reader)

        header = reader[1]
        header = [h[1:] for h in header]
        print(header)

        data = np.array(reader[6:]).astype(float)

        ans = {}
        for i, name in enumerate(header):
            t = data[:, 2*i]
            v = data[:, 2*i+1]
            ans[name] = scipy.interpolate.interp1d(t, v)

    return ans

def find_edges(dataset, level, number, rising):
    return fault.domain_read.find_edge_spice(
        dataset.x, dataset.y, 0, level,
        forward=True, rising=rising, count=number)


data = read_vcsv(filename)
print('hi')


N = 99

clock_edges = find_edges(data['clk_v2t'], 0.4, N, False)

def get_pulse_widths(dataset):
    rising  = find_edges(dataset, 0.4, N, True)
    falling = find_edges(dataset, 0.4, N, False)

    diff = []
    for r, f in zip(rising, falling):
        diff.append(f - r)
    diff = np.array(diff)
    return diff

def get_values(dataset):
    values = []
    for clk in clock_edges:
        values.append(dataset(clk))
    values = np.array(values)
    return values

pw_pos = get_pulse_widths(data['v2t_out_p'])
pw_neg = get_pulse_widths(data['v2t_out_n'])
pw_diff = pw_pos - pw_neg

v_pos = get_values(data['VinP'])
v_neg = get_values(data['VinN'])
v_diff = v_pos - v_neg

pos_slopes = []
for clk in clock_edges:
    dt = 0.05e-9 # 1e-11
    a = float(data['VinP'](clk - dt))
    b = float(data['VinP'](clk + dt))
    slope = (b - a) / (2*dt)
    pos_slopes.append(slope)
pos_slopes = np.array(pos_slopes)

up = pos_slopes > 0
down = pos_slopes < 0

#plt.plot(pos_slopes, '-+')
#plt.show()

x = v_diff
y = pw_diff
plt.plot(x[up], y[up], '+')
plt.plot(x[down], y[down], '*')
plt.grid()
plt.show()




print('bye')