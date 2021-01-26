import os
from collections import defaultdict
import shutil
def pad2File(num):
    _len = len(str(num))
    zero = '0'*(5-_len)
    return prefix+zero+str(num)+'.JPEG'

gTruthPath = './ILSVRC2012_validation_ground_truth.txt'
valPath = './val'
c2fPath = 'c2f.txt'
prefix = 'ILSVRC2012_val_000'

# build class to file dictionary
class_to_file_idx = defaultdict(list)
with open(gTruthPath, 'r') as f:
    file_idx = 1
    while (line:=f.readline()):
        class_to_file_idx[int(line)].append(file_idx)
        file_idx += 1

# write to txt 
with open(c2fPath, 'w') as f:
    for _class, file_idxs in sorted(class_to_file_idx.items(), key=lambda t:t[0]):
        file_idxs = map(pad2File,file_idxs)
        files = ','.join(file_idxs) +'\n'
        f.write(files)

# move files
with open(c2fPath, 'r') as f:
    _class = 1
    while (line:=f.readline()):
        files = line.split(',')
        if not os.path.exists('val/{}/'.format(_class)):
            os.makedirs('val/{}/'.format(_class))
        for fi in files:
            fi = fi.strip()
            oldPath = './val/{f}'.format(f=fi)
            newPath = './val/{c}/{f}'.format(c=_class, f=fi)
            newPath = newPath.replace('JPEG','jpeg')
            os.rename(oldPath, newPath)
        _class += 1