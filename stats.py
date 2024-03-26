import math 
import compression
import os
import glob
# import statistics
import numpy as np
import string
from custom_thread import custom_thread
from tqdm import tqdm

def std_deviation(l, l_median): 
    runs, n1, n2 = 0, 0, 0
      
    # Checking for start of new run 
    for i in range(len(l)): 
          
        # no. of runs 
        if (l[i] >= l_median and l[i-1] < l_median) or (l[i] < l_median and l[i-1] >= l_median): 
            runs += 1  
          
        # no. of positive values 
        if(l[i]) >= l_median: 
            n1 += 1   
          
        # no. of negative values 
        else:
            n2 += 1   
  
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2)) / (((n1+n2)**2)*(n1+n2-1))) 
  
    z = (runs-runs_exp)/stan_dev 
  
    return z


buffer = 'buffer'
storage = 'storage'

cmp = compression.Compression()
cmp.decompress_candidates(buffer, storage)


files_path = os.path.join(storage, '*')
files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
file_name = files[0]

l = open(file_name, 'r').read()
l = l.splitlines()


alphanum = sorted(set(string.ascii_letters + '2, 3, 4, 5, 6, 7'))
char_to_ind = {u: i for i, u in enumerate(alphanum)}


def encode_alphanum_strings(strings_list):
    encoded_list = np.empty((len(strings_list), 56), dtype=np.int32)

    for u, i in enumerate(strings_list):
        arr = np.array([char_to_ind[c] for c in i])
        encoded_list[u] = arr

    return encoded_list


encoded_list = encode_alphanum_strings(l)
print(encoded_list)

# @vectorize(['float32(float32, int32)'], target='cuda')
def mean_abs_deviation(s, A, encoded_list):
    mean_abs_devs = np.empty((1, 1), dtype=np.float32)
    
    for i in encoded_list:
        if not (i==A).all():
            av = np.absolute(i - A)
            s = s + av
    
    scattered_amd = s / len(encoded_list)**2
    mean_abs_devs[0] = np.sum(scattered_amd)

    return s, mean_abs_devs


def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

chunks = split_given_size(encoded_list, 50) # -> 200 cpu threads
# parts = encoded_list.shape[0] / 10

mean_abs_devs = np.empty((1, 1), dtype=np.float32)
s = 0

for chunk in tqdm(chunks):
    threads = []

    for i in chunk:
        thread = custom_thread(target=mean_abs_deviation, args=(s, i, encoded_list))
        threads.append(thread)
        thread.start()

    for thread in threads:
        s, A_amd = thread.join()
        np.concatenate((mean_abs_devs, A_amd), axis=0)


print(f'{mean_abs_devs}')
print(f'{mean_abs_devs.shape}')

# 149 samples -> max(mean_abs_devs)=array([1.8597211], dtype=float32) min(mean_abs_devs)=array([1.5139116], dtype=float32)
# 14979 samples -> max(mean_abs_devs)=array([4.e-45], dtype=float32) min(mean_abs_devs)=array([4.e-45], dtype=float32)

print(f'{max(mean_abs_devs)=}', f'{min(mean_abs_devs)=}')

# l_median= statistics.median(l)
# # print('l_median:', l_median)
# Z = abs(runsTest(l, l_median))
# print('Z-statistic= ', Z)

files = [file for file in files if not os.path.isdir(file)]

for file in files:
    os.remove(file)

