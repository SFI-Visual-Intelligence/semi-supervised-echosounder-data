import numpy as np

def zip_img_label(img_tensors, labels):
    img_label_pair = []
    for i, zips in enumerate(zip(img_tensors, labels)):
        img_label_pair.append(zips)
    print('num_pairs: ', len(img_label_pair))
    return img_label_pair

def flatten_list(nested_list):
    flatten = []
    for list in nested_list:
        flatten.extend(list)
    return flatten

def rebuild_input_patch(input_tensors_te, indim=32, outdim=256):
    # inp.shape = (N * 64, 4, 32, 32)
    # out.shape = (N, 4, 256, 256)
    # order: 0 - 7 // 8 - 15 ...
    N = len(input_tensors_te)//64
    inp_res = np.reshape(input_tensors_te, (N, 64, 4, 32, 32))
    patch_per_col = outdim // indim
    reshaped_te = []
    for inp in inp_res:
        for rowidx in range(patch_per_col):
            rowcon = np.concatenate(inp[rowidx * patch_per_col: (rowidx + 1)*patch_per_col], axis=-1)
            if rowidx == 0:
                colcon = rowcon
            else:
                colcon = np.concatenate([colcon, rowcon], axis=1)
        reshaped_te.append(colcon)
    return reshaped_te

def rebuild_pred_patch(inp, patch_len=32, outdim=256):
    count_patch = outdim//patch_len
    N = len(inp) // count_patch ** 2
    if len(np.shape(inp)) == 2:
        inp_sqr = np.reshape(inp, (N, count_patch, count_patch, 3))
        out_sqr = np.zeros((N, outdim, outdim, 3))
    elif len(np.shape(inp)) == 1:
        inp_sqr = np.reshape(inp, (N, count_patch, count_patch))
        out_sqr = np.zeros((N, outdim, outdim))

    for n, (in_one_sqr, out_one_sqr) in enumerate(zip(inp_sqr, out_sqr)):
        for row in range(8):
            for col in range(8):
                dupl = np.tile(in_one_sqr[row][col], (patch_len, patch_len)).reshape(patch_len, patch_len, -1)
                dupl = np.squeeze(dupl)
                out_one_sqr[row * patch_len: (row + 1) * patch_len, col * patch_len: (col + 1) * patch_len] = dupl
    return out_sqr
