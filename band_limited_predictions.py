import scipy.fft
import numpy as np

dt = 1

def pretty(Xs):
    N = len(Xs)
    fs = np.arange(N)*1/(N*dt)
    for f, X in zip(fs, Xs):
        print(f'{f:.4f}:\t{X:.4f}')
    print()

#xs = [0, 1, 2, 0, 0, 0, 2, 1]
#
#Xs = scipy.fft.fft(xs)
#
#pretty(Xs)




N = 16
fs_low = np.array([0, 1+0j])
print(fs_low)
Xs = np.zeros(N, dtype=fs_low.dtype)
Xs[0:len(fs_low)] = fs_low
Xs[-len(fs_low)+1:] = fs_low[:0:-1].conj()
print(Xs)
xs = scipy.fft.ifft(Xs)

pretty(xs)

xs_short = xs[:N//2]
Xs_short = scipy.fft.fft(xs_short)
pretty(Xs_short)
