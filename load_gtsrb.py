# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import numpy as np
import cv2
import os.path

def load_gtsrb_images(dataset_path, classes = range(43), max_num_imgs_per_class=float('Inf') ):
    
    img_size = [32, 32]
    image_block = np.zeros( img_size + [3, 0] )
    labels = np.zeros( 0, dtype=np.int32 )
    sign_ids = np.zeros( 0 )
    sign_id = 0
        
    for class_idx in classes:
        image_block_class = np.zeros( img_size + [3, 0] )
        
        dump_file = ("gtsrb_%02d.dump" % class_idx)
        if not os.path.exists(dump_file):
            
            seq_idx = 0
            sub_path = dataset_path + ("/%05d/"  % class_idx)
            next_image = None
            img_idx = 1
            img_cnt = 0
            while (img_idx != 0):
                img_idx = 0
                sign_id = sign_id + 1
                while (True):
                    image_file = sub_path + ("%05d_%05d.ppm"  % (seq_idx, img_idx) )
                    print("Reading file: " + image_file) 
                    next_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                    img_cnt += 1
                    print(img_cnt)
                    if not next_image is None:
                        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
                        next_image = cv2.resize(next_image, tuple(img_size), interpolation = cv2.INTER_CUBIC)
                        image_block_class = np.concatenate( [image_block_class, np.expand_dims(next_image, 3)], axis=3)
                        sign_ids = np.concatenate( [sign_ids, [sign_id] ], axis=0 )
                    else:
                        break
                    img_idx += 1
                seq_idx += 1
                
            print("Serializing images", image_block_class.shape)
            file = open(dump_file, 'wb')
            pickle.dump(image_block_class, file)
            file.close()

    class_descs = []
    class_descs.append('Speed limit 20') #0
    class_descs.append('Speed limit 30')
    class_descs.append('Speed limit 50') #2
    class_descs.append('Speed limit 60') #3
    class_descs.append('Speed limit 70') #4
    class_descs.append('Speed limit 80') #5
    class_descs.append('Derestriction 80') #6 
    class_descs.append('Speed limit 100')
    class_descs.append('Speed limit 120') #8
    class_descs.append('Prohibit Overtaking')
    class_descs.append('Prohibit Overtaking for trucks')
    class_descs.append('Right of way on next intersection')
    class_descs.append('Right of way on this street') #12 
    class_descs.append('Yield way') #13
    class_descs.append('Stop') #14
    class_descs.append('No entry') #15
    class_descs.append('No entry for trucks') #16
    class_descs.append('One way street') #17
    class_descs.append('Danger')
    class_descs.append('Attention road curves left') #19
    class_descs.append('Attention road curves right') #20
    class_descs.append('Attention S curve')
    class_descs.append('Attention bumpy road') #22
    class_descs.append('Attention slippery road')
    class_descs.append('Attention road will narrow') #24
    class_descs.append('Attention construction site')
    class_descs.append('Attention traffic lights') #26
    class_descs.append('Attention pedestrians')
    class_descs.append('Attention playing children') #28
    class_descs.append('Attention bicycle')
    class_descs.append('Attention snowfall') #30
    class_descs.append('Attention deer crossing')
    class_descs.append('Derestriction') #32
    class_descs.append('Turn right')
    class_descs.append('Turn left') #34
    class_descs.append('Forward')
    class_descs.append('Forward or right') #36
    class_descs.append('Forward or left')
    class_descs.append('Pass right') #38
    class_descs.append('Pass left')
    class_descs.append('Roundabout') #40
    class_descs.append('Derestriction overtaking')
    class_descs.append('Derestriction overtaking for trucks') #42

    for class_idx in classes:
        dump_file = ("gtsrb_%02d.dump" % class_idx)
            
        file = open(dump_file, 'rb')
        image_block_class = pickle.load(file)
        file.close()
        img_idxs = np.arange(image_block_class.shape[3])
        np.random.shuffle(img_idxs)
        image_block = np.concatenate( [image_block, image_block_class[:,:,:,img_idxs[0:max_num_imgs_per_class]] ], axis=3)
        labels = np.concatenate( [labels, [class_idx] * len(img_idxs[0:max_num_imgs_per_class])], axis=0 )
        
    image_block = np.transpose( image_block, axes=[3, 0, 1, 2])
                
    return image_block, labels, class_descs, sign_ids

if __name__ == '__main__':
    dataset_path = r'd:\Bochum\Lehre\CV-Praktikum Ueberarbeitung WS2017\Dataset\GTSRB\Final_Training\Images/'
    
    classes = range(43) 
    
    [imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, max_num_imgs_per_class=500)    