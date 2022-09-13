import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class ImageData(Dataset):
    def __init__(self, data_path, classes=1000):
        self.data_path = data_path

        self.image_names = []

        for file in os.listdir(data_path):
            self.image_names.append(file)

        self.N = len(self.image_names)

        self.permutations = self.__retrive_permutations(classes)

        self.__image_transformer = T.Compose([
            T.Resize(256, T.InterpolationMode.BILINEAR),
            T.CenterCrop(255)])
        self.__augment_tile = T.Compose([
            T.RandomCrop(64),
            T.Resize((75, 75), T.InterpolationMode.BILINEAR),
            T.Lambda(self.rgb_jittering),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image_name = self.data_path + '/' + self.image_names[index]

        img = Image.open(image_name).convert('RGB')

        img.save("original.jpeg")

        if img.size[0] != 255:
            img = self.__image_transformer(img)
        
        img.save("cropped.jpeg")


        s = float(img.size[0]) // 3
        a = s // 2
        tiles = [None] * 9
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # tile.save(f"tile{n}.jpeg")

            # # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = T.Normalize(mean=m.tolist(), std=s.tolist())
            # tile = norm(tile)
            tiles[n] = tile
            

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def __retrive_permutations(classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    @staticmethod
    def rgb_jittering(im):
        im = np.array(im, 'int32')
        for ch in range(3):
            im[:, :, ch] += np.random.randint(-2, 2)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype('uint8')


import matplotlib.pyplot as plt
if __name__ == '__main__':
    data_path = 'data/ILSVRC2012_img_train'
    dataset = ImageData(data_path)
    permuted,order,original  = dataset[0]
    print("Dataset Permutation : ",dataset.permutations[order])
    transform=T.ToPILImage()
    f, ax = plt.subplots(3, 3)
    for i,tile in enumerate(original):
        tile=transform(tile)
        ax[i//3][i%3].imshow(tile)
        # tile.save(f"original{i}.jpeg")
    plt.savefig("orig.jpeg")

    f, ax = plt.subplots(3, 3)
    for i,tile in enumerate(permuted):
        tile=transform(tile)
        ax[i//3][i%3].imshow(tile)
        # tile.save(f"permuted{i}.jpeg")
    plt.savefig("perm.jpeg")
    




