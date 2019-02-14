import h5py
f = h5py.File("weights.h5")
list(f)
#['activation_1', 'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'dropout_1', 'dropout_2', 'fc6', 'fc7', 'fc8', 'flatten_1', 'input_1', 'maxpooling2d_1', 'maxpooling2d_2', 'maxpooling2d_3', 'maxpooling2d_4', 'maxpooling2d_5', 'zeropadding2d_1', 'zeropadding2d_10', 'zeropadding2d_11', 'zeropadding2d_12', 'zeropadding2d_13', 'zeropadding2d_2', 'zeropadding2d_3', 'zeropadding2d_4', 'zeropadding2d_5', 'zeropadding2d_6', 'zeropadding2d_7', 'zeropadding2d_8', 'zeropadding2d_9']

f['fc7']
#<HDF5 group "/fc7" (2 members)>
list(f['fc7'])
#['fc7_W', 'fc7_b']
f['fc7']['fc7_W'].shape
#(4096, 4096, 1, 1)
w=f['fc7']['fc7_W']

import numpy as np
w=np.array(f['fc7']['fc7_W'])
w.shape
#(4096, 4096, 1, 1)
np.save('fc7_w.npy', w)
