import cv2
import numpy as np
import tensorflow as tf
import unicodedata
import multiprocessing

class ImageDataset():
    """Face Landmarks dataset."""

    def __init__(self, data_path, char_path, batch_size=10, training=True, transform=None):
        """
        Args:
            data_path (string): Path to the files with images and their annotations.
            length (string): image number.
            class_num (int): class number.
        """
        with open(data_path) as fh:
            self.img_and_label = fh.readlines()
        with open(char_path) as f:
            self.char_id_map = {char.strip():idx for idx,char in enumerate(f)}
            print (self.char_id_map)
        self.length = len(self.img_and_label)
        self.class_num = len(self.char_id_map)
        self.indexes = np.arange(self.length)
        self.batch_size = batch_size
        self.transform = transform if training else None
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_and_label = self.img_and_label[index].strip()
        pth, word = img_and_label.split(' ') # image path and its annotation

        image = cv2.imread(pth)#,0)
        image = cv2.pyrDown(image).astype('float32') # 100*100
        
        word = [ord(var)-97 for var in word] # a->0

        label = np.zeros((self.class_num)).astype('float32')

        for ln in word:
            label[int(ln+1)] += 1 # label construction for ACE

        label[0] = len(word)
        return pth,image,label

 
    def data_generation(self):
        steps_of_per_epoch = self.length//self.batch_size
        while True:
            with multiprocessing.Pool(processes=8) as pool:
                np.random.shuffle(self.img_and_label)
                for i, data in enumerate(range(steps_of_per_epoch)):
                    paths = []
                    images = []
                    labels = []
                    index_batch = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
                    for idx in index_batch:
                        # print (idx)
                        path,image,label = self.__getitem__(idx)
                        # print ("label", label)
                        paths.append(path)
                        images.append(image)
                        labels.append(label)

                    yield {"path":np.array(paths), "images": np.array(images)}, np.array(labels)


