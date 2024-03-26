import math 
import compression
import os
import glob
import cupy as np
import string

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
def mean_abs_deviation(mean_abs_devs, encoded_list):
    for u, A in enumerate(encoded_list):
        sum = 0
    
        for i in encoded_list:
            if not (i==A).all():
                av = np.absolute(i - A)
                sum = sum + av
        
        scattered_amd = sum / len(encoded_list)**2
        mean_abs_devs[u] = np.sum(scattered_amd)

    return mean_abs_devs


mean_abs_devs = np.empty((encoded_list.shape[0], 1), dtype=np.float32)
mean_abs_devs = mean_abs_deviation(mean_abs_devs, encoded_list)

print(f'{max(mean_abs_devs)=}', f'{min(mean_abs_devs)=}')


files = [file for file in files if not os.isdir(file)]

for file in files:
    os.remove(file)

