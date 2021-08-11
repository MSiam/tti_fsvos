import numpy as np
import os


ignore_seqs = ["1146_lps8_r-6J00", "118_fgmRMeHommU", "1278_C4zCNFn1xIs",
               "1275_ARcg-EyKWrA", "1302_OCLlE02BHGk", "131_eUiWFntut00",
               "1339_J73l0QCY8YM", "1353_CszoyQ3HMcM", "1476_75AL-XN84cI",
               "1887_tUt0N6eGtGY", "198_b8euyKNT2wY", "1026_kJ_8F7YIEg4",
               "1265_PBn1W-aOFUA", "1035_248bbw7mpdw", "1107_YXyd44eY_VY",
               "1154_B4zEa_7Ejtk", "1282_cSmDnZFwqIM", "1477_GkuOGCQiUlk",
               "2051_e0EI-QqHPIA"]

main_dir = "../../lists/vspw/"
total = 0
total_frames = 0
for fname in os.listdir(main_dir):
    seqs = []
    with open(os.path.join(main_dir, fname), 'r') as f:
        for line in f:
            seqs.append(line.split("/")[1])
            if seqs[-1] not in ignore_seqs:
                total_frames += 1
        total += len(set(seqs))

print(total - len(ignore_seqs))
print(total_frames)
