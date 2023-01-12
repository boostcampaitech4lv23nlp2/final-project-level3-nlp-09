from .crop_data import CropDataset
from .seg_data import SegDataset

from sodiumData.utils.io import SingleHubAPI

from typing import Optional, Callable


def autodataset(data_name: str, 
                target_type: str, 
                valid_ratio: float, 
                root: str = '/data/.hubAPI', 
                transform: Optional[Callable] = None,
                del_labels=['__ignore__', '_ignore_', 'mix-food', 'food', '그릇', 'remove'],
                keep_labels=None,
                old_labels=None,
                new_labels=None):

    hub = SingleHubAPI.get_api()
    dataset = hub.get_dataset(name=data_name,
                                   type=target_type,
                                   download=True,
                                   folder=root)
    dataset.split(valid_ratio)
    dataset.update_categories(keep_labels=list(keep_labels) if keep_labels else None,
                                new_labels=list(new_labels) if new_labels else None,
                                old_labels=list(old_labels) if old_labels else None,
                                del_labels=list(del_labels) if del_labels else None,
                                load_dicts=False)

    if target_type == 'crop':
        return CropDataset(dataset.train_path, transform=transform), \
                CropDataset(dataset.test_path, transform=transform)
    else:
        return SegDataset(dataset.train_path, transform=transform), \
                SegDataset(dataset.test_path, transform=transform)

