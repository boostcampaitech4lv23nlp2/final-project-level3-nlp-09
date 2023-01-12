import cv2
import numpy as np
import os.path as osp

from nuviAug.geometric.helpers import Resize


class CropNearBBox:
    def __init__(self,
                 min_box=1,
                 max_box=None,
                 max_margin=0.3,
                 min_box_size=10,
                 p=0.5
     ):
        self.p = p
        self.min_box = min_box
        self.max_box = max_box
        self.max_margin = max_margin
        self.min_box_size = min_box_size
        self.resizer = Resize()

    def __call__(self, img, bboxes, labels, masks, dsize=None) -> dict:
        if np.random.random(1) > self.p:
            return dict(image=img, bboxes=bboxes, labels=labels)

        im_h, im_w, _ = img.shape
        max_box = len(bboxes) if self.max_box is None else min(self.max_box, len(bboxes))
        n_instance = np.random.randint(self.min_box, max_box) if max_box > self.min_box else self.min_box
        if n_instance == 0:
            return dict(image=img, bboxes=bboxes, labels=labels)

        if isinstance(bboxes, list):
            bboxes = np.array(bboxes).astype(np.float32)
        if isinstance(labels, list):
            labels = np.array(labels)

        inst_index = np.random.choice(range(len(bboxes)),
                                    size=n_instance, replace=False)

        picked_bboxes = bboxes[inst_index]
        w, h = bboxes[:, 2]-bboxes[:, 0], bboxes[:, 3]-bboxes[:, 1]

        x_ratio = 1 + np.random.uniform(0.1, self.max_margin, size=1)
        y_ratio = 1 + np.random.uniform(0.1, self.max_margin, size=1)
        x1 = max(0, (picked_bboxes[:, 0] - w * x_ratio).min())
        y1 = max(0, (picked_bboxes[:, 1] - h * y_ratio).min())
        x2 = min(im_w, (picked_bboxes[:, 2] + w * x_ratio).max())
        y2 = min(im_h, (picked_bboxes[:, 3] + h * y_ratio).max())

        image = img[int(y1):int(y2), int(x1):int(x2)]
        cropped_h, cropped_w = image.shape[:2]
        if cropped_h == 0 or cropped_w == 0:
            image = img
            picked_masks = masks
            picked_bboxes = bboxes
            picked_labels = labels
        else:
            picked_bboxes = bboxes.copy()
            picked_bboxes[:, 0::2] -= x1
            picked_bboxes[:, 1::2] -= y1

            keep = np.array([False] * len(bboxes))
            keep[bboxes[:, 0] >= x1] = True
            keep[bboxes[:, 2] <= x2+x1] = True
            keep[bboxes[:, 1] >= y1] = True
            keep[bboxes[:, 3] <= y2+y1] = True

            picked_labels = labels[keep]
            picked_bboxes = picked_bboxes[keep]
            picked_masks = [mask[int(y1):int(y2), int(x1):int(x2)] for mask in masks[keep]] if masks else masks

        if dsize is not None:
            image, bboxes, masks, _ = self.resizer(image, picked_bboxes, picked_masks,
                                                   size=(int(x2-x1), int(y2-y1)),
                                                   dsize=dsize)

        keep = (picked_bboxes[:, 2] - picked_bboxes[:, 0]) \
               * (picked_bboxes[:, 3] > picked_bboxes[:, 1]) >= self.min_box_size
        picked_bboxes = picked_bboxes[keep]
        picked_masks = picked_masks[keep] if masks else masks
        picked_labels = picked_labels[keep]

        return dict(
            image=image,
            masks=picked_masks,
            bboxes=picked_bboxes,
            labels=picked_labels
        )


class MosaicAugmentation:
    def __init__(self,
                 root,
                 coco,
                 image_ids,
                 do_seg=False,
                 p=1.0,
    ):
        self.p = p
        self.root = root
        self.coco = coco
        self.data = image_ids
        self.do_seg = do_seg
        self.cropper = CropNearBBox(min_box=1,
                                    max_box=3,
                                    max_margin=0.15,
                                    p=1.0)

    def get_centre(self, width, height):
        x = int(np.random.uniform(width * 0.3, width * 0.8))
        y = int(np.random.uniform(height * 0.3, height * 0.8))
        return x, y

    def get_samples(self, index):
        props = [1.0/(len(self.data)-1)] * len(self.data)
        props[index] = 0.0
        indicies = np.random.choice(range(len(self.data)), size=3,
                                    replace=False, p=props)
        return indicies

    def get_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        img = cv2.imread(path)
        h, w, _ = img.shape
        return img, (w, h), file_name

    def get_annots(self, image_id, annot_ids):
        target = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]
        masks = [self.coco.annToMask(obj).reshape(-1) for obj in target
                                    if obj['segmentation'] is not None]
        if masks:
            masks = np.vstack(masks)
        bboxes = np.array([obj['bbox'] for obj in target]).astype(np.float32)
        labels = np.array([obj['category_id'] for obj in target])
        bboxes[:, 2:] += bboxes[:, :2]
        return bboxes, masks, labels

    def get_data(self, idx):
        img_id = self.data[idx]
        img, (w, h), file_name = self.get_image(img_id)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        bboxes, masks, labels = self.get_annots(img_id, ann_ids)
        if len(masks) > 0 and self.do_seg:
            masks = masks.reshape(-1, h, w)
        elif self.do_seg:
            masks = np.zeros((len(bboxes), h, w), dtype=np.uint8)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
        return img, bboxes, masks, labels, file_name, img_id, (w, h)

    def __call__(self, index):
        if np.random.random() > self.p:
            return None

        img, bboxes, masks, labels, file_name, img_id, (w, h) = self.get_data(index)

        #TODO: remove try except... to tired to do right now
        try:
            ctr_x, ctr_y = self.get_centre(w, h)
            mosaic_image = np.full(img.shape, 255, dtype=np.uint8)
            valid_masks = np.zeros((1, h, w), dtype=np.uint8)

            # top left
            cropped = self.cropper(img, bboxes, labels, masks, dsize=(ctr_x, ctr_y))
            img = cropped['image']
            valid_bboxes = cropped['bboxes']
            valid_labels = cropped['labels']
            if self.do_seg:
                valid_masks[:, :ctr_y, :ctr_x] = cropped['masks']
            mosaic_image[:ctr_y, :ctr_x] = img

            add_index = self.get_samples(index)
            for i, idx in enumerate(add_index):
                img, bboxes, masks, labels, file_name, img_id, _ = self.get_data(idx)

                valid_mask = np.zeros((1, h, w), dtype=np.uint8)
                if i == 0:
                    # top right
                    cropped = self.cropper(img, bboxes, labels, masks, dsize=(int(w-ctr_x), ctr_y))
                    img = cropped['image']
                    bbox_move = np.array([ctr_x, 0, ctr_x, 0])
                    if self.do_seg:
                        valid_mask[:, :ctr_y, ctr_x:] = cropped['masks']
                    mosaic_image[:ctr_y, ctr_x:] = img
                elif i == 1:
                    # bottom left
                    cropped = self.cropper(img, bboxes, labels, masks, dsize=(ctr_x, int(h-ctr_y)))
                    img = cropped['image']
                    bbox_move = np.array([0, ctr_y, 0, ctr_y])
                    if self.do_seg:
                        valid_mask[:, ctr_y:, :ctr_x] = cropped['masks']
                    mosaic_image[ctr_y:, :ctr_x] = img
                elif i == 2:
                    # bottom right
                    cropped = self.cropper(img, bboxes, labels, masks, dsize=(int(w-ctr_x), int(h-ctr_y)))
                    img = cropped['image']
                    bbox_move = np.array([ctr_x, ctr_y, ctr_x, ctr_y])
                    if self.do_seg:
                        valid_mask[:, ctr_y:, ctr_x:] = cropped['masks']
                    mosaic_image[ctr_y:, ctr_x:] = img

                valid_masks = np.append(valid_masks, valid_mask, axis=0)
                valid_bboxes = np.append(valid_bboxes, cropped['bboxes'] + bbox_move)
                valid_labels = np.append(valid_labels, cropped['labels'])

            return dict(image=mosaic_image,
                        masks=valid_masks if self.do_seg else np.zeros((0, h, w), np.uint8),
                        bboxes=valid_bboxes.reshape(-1, 4),
                        labels=valid_labels,
                        img_id=img_id,
                        file_name=file_name,
                        shape=(w, h))
        except Exception as e:
            return None
