from datasets import load_dataset


class FoodDataset:
    def __init__(self, preprocess):
        self.dataset = load_dataset("food101")
        self.dataset.set_transform(self.transform_func)
        self.preprocess = preprocess

    def get_train_dataset(self):
        return self.dataset["train"]

    def transform_func(self, examples):
        examples["image"] = [self.preprocess(image) for image in examples["image"]]
        return examples
