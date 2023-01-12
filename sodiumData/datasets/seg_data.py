import cv2
import numpy as np
import os.path as osp

from pathlib import Path
from pycocotools.coco import COCO
from typing import Optional, Callable, Union
from torch.utils.data import Dataset


class SegDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the  # type: str
                                        raw images`
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 transform: Optional[Callable] = None,
                 do_seg: bool = True,
                 has_gt: bool = True,
                 **kwargs):
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path).resolve()

        self.data_path = data_path
        self.root_path = data_path.parent
        self.coco = COCO(annotation_file=data_path)
        self.classes = self.coco.cats

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        self.ids = self.ids
        self.transform = transform
        self.mosaic_tfms = None
        # self.mosaic_tfms = MosaicAugmentation(root_path, self.coco, self.ids, p=0.3)
        self.has_gt = has_gt
        self.do_seg = do_seg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_tensor, bboxes, masks, labels, img_id, file_name, shape = self.pull_item(index)
        return img_tensor, bboxes, masks, labels, img_id, file_name, shape

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
        """
        res = self.pull_data(index)
        img = res['image']
        masks = res['masks']
        bboxes = res['bboxes']
        labels = res['labels']
        img_id = res['img_id']
        file_name = res['file_name']
        width, height = res['shape']

        if self.transform is not None:
            if len(bboxes) > 0:
                transformed = self.transform(image=img, bboxes=bboxes, masks=masks, labels=np.array(labels))

                img = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['labels']
                masks = transformed['masks']
            else:
                transformed = self.transform(image=img, bboxes=[], masks=[])
                img = transformed['image']
                masks = np.zeros((0, height, width), dtype=np.float)
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ), dtype=np.int64)

        # Concerns: do we really need to ensure every sample has at least a box?
        # I believe if there is no boxes after transformation, it gives the model a negative sample.
        # if len(bboxes) == 0:
        #     # if there is no boxes
        #     return self.pull_item(random.randint(0, len(self.ids) - 1))

        return dict(
            img=img, bboxes=bboxes, masks=masks, lables=labels,
            img_id=img_id, file_name=file_name, shape=(height, width)
        )

    def pull_data(self, index: int) -> dict:
        img_id = self.ids[index]
        file_name = self.coco.loadImgs(index)[0]['file_name']
        img = cv2.imread(osp.join(self.root, file_name), cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        target = []
        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=index)
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]

        if len(target) > 0 and self.do_seg:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
        elif self.do_seg:
            masks = np.zeros((len(target), height, width))
        else:
            masks = np.zeros((0, height, width))

        bboxes = np.array([obj['bbox'] for obj in target]).astype(np.float32)
        labels = np.array([obj['category_id'] for obj in target])
        bboxes[:, 2:] += bboxes[:, :2]
        return img, bboxes, masks, labels, img_id, file_name, (height, width)
