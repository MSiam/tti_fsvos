import os

src_dir = '/home/msiam/Dataset/TAO/frames/'
target_dir = '/media/msiam/Elements/TAOVOS-5i/frames/'

with open('../../lists/taovos/pascal/seqs.txt', 'r') as f:
    for line in f:
        sq_path = line.strip()
        target = os.path.join(target_dir, sq_path)
        if not os.path.exists(target):
            os.makedirs(target)

        os.system('cp -r '+os.path.join(src_dir, sq_path)+' '+os.path.join(target_dir, sq_path))
