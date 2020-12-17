import os
import shutil
import numpy as np

def grayscale(s):
    return np.dot(s[..., :3], [0.299, 0.587, 0.114])

def clear_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def convert_action(a, reverse=False):
    if reverse:
        return np.array([8, 4, 2, 1]) @ a
    l = [int(i) for i in f"{a:08b}"[-4:]]
    return np.array(l, dtype=np.int8)

# def convert_action(a, reverse=False):
#     if reverse:
#         x = 4
#         for i, j in enumerate(a):
#             if j == 1:
#                 x = i
#                 return x
#         return x
#     l = [0, 0, 0, 0]
#     if a < 4:
#         l[a] = 1
#     return np.array(l, dtype=np.int8)