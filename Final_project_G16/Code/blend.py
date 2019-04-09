import cv2
import os
import sys
import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import concurrent.futures
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
        max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target


def test(path, dir_name1,dir_name2):
    img_mask = np.asarray(PIL.Image.open('./test1_mask5.png'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open('./'+dir_name1+'/'+path))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open('./'+dir_name2+'/'+path)) #13312_12800 (1)
    img_target.flags.writeable = True
   # print(len(img_target))
   # print(len(img_source))
    img_ret = blend(img_target, img_source, img_mask, offset=(0,0))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))
    img_ret.save('./result/'+path)
    print(path)
   
    
    
    
'''def stitch_image(imgs_path, dir_name):
    imgs = [cv2.imread(dir_name + '/' + path) for path in imgs_path]
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for item in number_list:
            executor.submit(evaluate_item,  item)    '''



if __name__ == '__main__':
    argc = len(sys.argv)
    if(argc != 3):
        print('[usage]: python3 %s' % sys.argv[0])
        exit(0)        
    dir_name1 = argv[1]
    dir_name2 = argv[2]
    imgs_path1 = os.listdir(dir_name1)
    imgs_path2 = os.listdir(dir_name2)
    it=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: 
        futures= [executor.submit(test, path, dir_name1,dir_name2) for path in imgs_path1]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())        
       # for path in imgs_path1:
       #     print(it,path)
       #     ++it
       #     executor.submit(test, path, dir_name1,dir_name2)
        ##test(path,dir_name1,dir_name2)
   # stitched = stitch_image(imgs_path, dir_name)
   # cv2.imwrite('reconstructed.jpg', np.array(stitched))