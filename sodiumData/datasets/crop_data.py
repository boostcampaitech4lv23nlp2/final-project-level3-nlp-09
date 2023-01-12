import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Union, Optional, Callable

from sodiumData.utils.io import read_json


class CropDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 transform: Optional[Callable] = None,
                 *args, **kwargs
    ) -> None:
        super(CropDataset, self).__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path).resolve()

        self.data_path = data_path
        self.root_path = data_path.parent
        self.transform = transform
        json_data = read_json(data_path)

        self.samples = []
        self.catId_to_idx = {}
        self.id_to_cls = {}
        for idx, category in enumerate(json_data['categories']):
            self.catId_to_idx[category['id']] = idx
            self.id_to_cls[idx] = category['label']

        for i, image in enumerate(json_data['images']):
            file_name = image['file_name']
            if file_name.startswith('/'):
                file_name = file_name[1:]

            path = self.root_path / file_name
            target = self.catId_to_idx[image['category_id']]
            self.samples.append((path, target))
        self.classes = list(self.id_to_cls.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        class_name = self.id_to_cls[target]
        basename = path.name

        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        sample = np.array(img, dtype=np.uint8)
        if self.transform is not None:
            transformed = self.transform(image=sample)
            sample = transformed['image']
            if 'token_mask' in sample:
                token_mask_tensor = transformed['token_mask']
            else:
                token_mask_tensor = None

        return dict(
            image=sample, token_mask=token_mask_tensor,
            target=target, index=index, basename=basename, class_name=class_name
        )
