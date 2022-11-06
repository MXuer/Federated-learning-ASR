import os
import torch 
from glob2 import glob
from os.path import basename, dirname, join, sep, abspath
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

def get_spk_id(path):
    sid = path.split(sep)
    sid = sid[-3]
    sid = int(sid.split("_")[1])
    return sid

def get_spk_utt_count(sid, data_root):
    wavscp = f"{data_root}/16/train/wav.scp"
    with open(wavscp, 'r') as wfile:
        utts = len(wfile.readlines())
    return utts

fl_root = sys.argv[1]
data_root = sys.argv[2]

input_models = list(glob(f'{fl_root}/**/train_*_ep*_ck*_*.pkl'))
epck = []
spk = []
utts = []

for mod in input_models:
    sid = get_spk_id(mod)
    spk += [sid]
    utts += [get_spk_utt_count(sid, data_root)]
    mod = basename(mod)
    mod = mod.split("_")[-3:-1]
    epck += [",".join(mod)]

# for x in epck:
#     print(x, epck[0])
# assert all(x==epck[0] for x in epck), 'Something messed up the training order. Please delete fl speaker-exp_files and restart.'

utts = np.array(utts)
total_utts = utts.sum()
weights = utts / total_utts
print(f"Calculating avg model @({epck[0]})")

avg_model_par = None
for ids, mod in enumerate(input_models):
    print(f"{ids} / {len(input_models)} | loading {mod}...")
    if avg_model_par == None:
        avg_model_par = torch.load(mod, map_location='cpu')['model_par']
        for key in avg_model_par.keys():
            avg_model_par[key] = weights[ids] * avg_model_par[key]
    else:
        _avg_model_par = torch.load(mod, map_location='cpu')['model_par']
        for key in avg_model_par.keys():
            avg_model_par[key] += weights[ids] * _avg_model_par[key]


def save_model(model_files):                                                          
    for ids, mod in enumerate(model_files):
        final_model_file = os.path.join(os.path.dirname(mod), "final_architecture1.pkl")
        print(f"{ids} / {len(input_models)} | loading {mod}...")
        if os.path.exists(final_model_file) and os.path.getsize(final_model_file) != 0:
            continue
        base_model = torch.load(mod, map_location='cpu')
        base_model['model_par'] = avg_model_par
        torch.save(base_model, final_model_file)


nparts = 1
nums_part = len(input_models) // nparts + 1
parts = [input_models[index:index + nums_part] for index in range(0, len(input_models), nums_part)]
all_tasks = []
with ThreadPoolExecutor(max_workers=nparts) as t:
    for part in parts:
        task_per = t.submit(save_model, part)
        all_tasks.append(task_per)
    wait(all_tasks, return_when=ALL_COMPLETED)


