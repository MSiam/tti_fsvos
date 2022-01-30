import os
import numpy as np

main_dir = '/local/data1/msiam/VSPW_480p/data/'

with open('lists/vspw/all.txt', 'w') as f:
    for seq in os.listdir(main_dir):
        for fname in os.listdir(os.path.join(main_dir, seq, 'origin')):
            final_fname = os.path.join('data', seq, 'mask', fname.replace('jpg', 'png'))
            f.write(final_fname + '\n')
