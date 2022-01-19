f = open('/home/dstanley/research/fixture/DelayTest_0_post_regression.pickle', 'rb')

import pickle
x = pickle.load(f)
print(x)
