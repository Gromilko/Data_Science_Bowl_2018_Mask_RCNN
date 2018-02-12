import os
import numpy as np
import skimage.io

def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten() == 1)[0]
    # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


PRED_DIR = '/home/futura/PycharmProjects/Kaggle/DSB_data/predict_stage1_test'

masks_dir = os.listdir(PRED_DIR)

with open('submission.csv', 'w') as f:
    f.write('ImageId,EncodedPixels\n')
    for count, id in enumerate(masks_dir):
        if count < 0:
            f.write('{},1 1\n'.format(id))
        else:
            print(count, id)
            MASKS_DIR = os.path.join(PRED_DIR, id, 'masks')
            masks = os.listdir(MASKS_DIR)

            find_prec = []
            for mask in masks:
                a = np.where(skimage.io.imread(os.path.join(MASKS_DIR, mask)) == 255, 1, 0)
                find_prec.append(a)
            find_prec = np.array(np.stack(find_prec))
            print('/', find_prec.shape)
            # print(')))', np.min(find_prec), np.max(find_prec))
            for i in range(find_prec.shape[0]-1):

                q = np.sum(find_prec[i+1:, :, :], 0)
                sm = np.where(q != 0, 1, 0)
                skimage.io.imsave('asd.png', sm*255)

                find_prec[i, :, :] -= sm
                find_prec[i, :, :] = np.where(find_prec[i, :, :] == 1, 1, 0)
                # print(np.min(find_prec[i, :, :]), np.max(find_prec[i, :, :]))
                rle = rle_encoding(find_prec[i, :, :])
                if len(rle) == 0:
                    print('\nrle=0!', id, '\n')
                    skimage.io.imsave('{}_{}.png'.format(id, i), find_prec[i, :, :] * 255)
                else:
                    rle = ' '.join([str(i) for i in rle])
                    f.write('{},{}\n'.format(id, rle))

            rle = ' '.join([str(i) for i in rle_encoding(find_prec[-1, :, :])])
            f.write('{},{}\n'.format(id, rle))

        # skimage.io.imsave('asd.png', np.sum(find_prec, 0)*255)