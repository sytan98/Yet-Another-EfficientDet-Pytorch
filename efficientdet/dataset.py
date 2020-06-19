import os
import torch
import numpy as np

from PIL import ImageEnhance, ImageFont, ImageDraw, Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from albumentations.pytorch import ToTensor
from albumentations import (
    BboxParams,
    HorizontalFlip,
    RandomSizedBBoxSafeCrop,
    Crop,
    Compose,
    Rotate
)


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img,h,w = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        h,w,_ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255. , h, w

    def load_annotations(self, image_index, h, w):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            if a['bbox'][0] < 0:
                print("off x min")
                a['bbox'][0] = 0
            if a['bbox'][1]< 0:
                print("off y min")
                a['bbox'][1] = 0
            if a['bbox'][0] + a['bbox'][2] > w:
                print("off x max")
                a['bbox'][2] = w - a['bbox'][0]
            if a['bbox'][1] + a['bbox'][3] > h:
                print("off y max")
                a['bbox'][3] = h - a['bbox'][1]
                
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


def aug_colorbalance(img, annots, color):
    if np.random.rand() < color:
        # print("random")
        img = Image.fromarray(img.astype(np.uint8))
        color_factors=[0.5,1.5]
        factor = color_factors[0] + np.random.uniform() * (color_factors[1] - color_factors[0])
        enhancer = ImageEnhance.Color(img)
        img = np.array(enhancer.enhance(factor))
    return img, annots

def aug_contrast(img, annots, contrast):
    if np.random.rand() < contrast:
        # print("random")
        img = Image.fromarray(img.astype(np.uint8))
        contrast_factors=[0.5,1.5]
        factor = contrast_factors[0] + np.random.uniform() * (contrast_factors[1] - contrast_factors[0])
        enhancer = ImageEnhance.Contrast(img)
        img = np.array(enhancer.enhance(factor))
    return img, annots

def aug_brightness(img, annots, brightness):
    if np.random.rand() < brightness:
        # print("random")
        img = Image.fromarray(img.astype(np.uint8))
        brightness_factors=[0.5,1.5]
        factor = brightness_factors[0] + np.random.uniform() * (brightness_factors[1] - brightness_factors[0])
        enhancer = ImageEnhance.Brightness(img)
        img = np.array(enhancer.enhance(factor))
    return img, annots

def aug_sharpness(img, annots, sharpness):
    if np.random.rand() < sharpness:
        # print("random")
        img = Image.fromarray(img.astype(np.uint8))
        sharpness_factors=[0.5,5.0]
        factor = sharpness_factors[0] + np.random.uniform() * (sharpness_factors[1] - sharpness_factors[0])
        enhancer = ImageEnhance.Sharpness(img)
        img = np.array(enhancer.enhance(factor))
    return img, annots

def aug_horizontal_flip(image, annots, flip_x):
    if np.random.rand() < flip_x:
        # print("random")
        image = image[:, ::-1, :]

        rows, cols, channels = image.shape

        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()

        x_tmp = x1.copy()

        annots[:, 0] = cols - x2
        annots[:, 2] = cols - x_tmp

    return image, annots

def compute_reasonable_boundary(annots):
        xmin = min([bb[0] for bb in annots])
        xmax = max([bb[2] for bb in annots])
        ymin = min([bb[1] for bb in annots])
        ymax = max([bb[3] for bb in annots])
        return xmin, xmax, ymin, ymax

def aug_crop(img, annots, crop_p):
    if np.random.rand() < crop_p:
        # Compute bounds such that no boxes are cut out
        # print(annots)
        xmin, xmax, ymin, ymax = compute_reasonable_boundary(annots)
        # print(xmin, xmax, ymin, ymax)
        img = Image.fromarray(img.astype(np.uint8))
        W,H = img.size
        # print(W,H)
        # Choose crop_xmin from [0, xmin]
        crop_xmin = max( np.random.uniform() * xmin, 0 )
        # Choose crop_xmax from [xmax, 1]
        crop_xmax = min( xmax + (np.random.uniform() * (W-xmax)), W)
        # Choose crop_ymin from [0, ymin]
        crop_ymin = max( np.random.uniform() * ymin, 0 )
        # Choose crop_ymax from [ymax, 1]
        crop_ymax = min( ymax + (np.random.uniform() * (H-ymax)), H)
        # Compute the "new" width and height of the cropped image
        # crop_w = crop_xmax - crop_xmin
        # crop_h = crop_ymax - crop_ymin
        # print(crop_xmin, crop_xmax, crop_ymin, crop_ymax)
        cropped_labels = []
        for x1,y1,x2,y2,c in annots:
            x1_new = (x1 - crop_xmin)
            y1_new = (y1 - crop_ymin)
            x2_new = (x2 - crop_xmin)
            y2_new = (y2 - crop_ymin)
            cropped_labels.append( (x1_new, y1_new, x2_new, y2_new, c) )

        # Compute the pixel coordinates and perform the crop
        impix_xmin = int(crop_xmin)
        impix_xmax = int(crop_xmax)
        impix_ymin = int(crop_ymin)
        impix_ymax = int(crop_ymax)
        # print(impix_xmin, impix_ymin, impix_xmax, impix_ymax)
        img = np.array(img.crop((impix_xmin, impix_ymin, impix_xmax, impix_ymax)))
        annots = np.array(cropped_labels)

    return img, annots

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))

def aug_albumentations(image, annots):
    bb = []
    cat = []
    for i in annots:
      bb.append(i[:4].tolist())
      cat.append(i[4])
    category_id_to_name = {1:"tops", 2: "trousers", 3: "outerwear", 4: "dresses", 5: "skirts"}
    annotations1 = {'image': image, "bboxes": bb, 'category_id': cat}

    aug = get_aug([Rotate(p=0.5, limit = 15), RandomSizedBBoxSafeCrop(p=0.5, width = 1024, height = 1024)])
    augmented = aug(**annotations1)
    image = augmented['image']

    annotation_gather = []
    for idx, bbox in enumerate(augmented['bboxes']):
      x_min, x_max, y_min, y_max = bbox
      annotation_gather.append( [x_min, x_max, y_min, y_max, augmented['category_id'][idx]])
    annots = np.array(annotation_gather)
    
    return image, annots

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, p={'flip':0.5}):
        image, annots = sample['img'], sample['annot']
        image, annots = aug_horizontal_flip(image, annots,  p['flip'])
        image, annots = aug_albumentations(image, annots)
        # image, annots = aug_colorbalance(image, annots,  p['color'])
        # image, annots = aug_contrast(image, annots, p['contrast'])
        # image, annots = aug_brightness(image, annots,p['brightness'])
        # image, annots = aug_sharpness(image, annots, p['sharpness'])
        
        sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
