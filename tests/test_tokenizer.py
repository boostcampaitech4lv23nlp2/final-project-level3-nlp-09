import json
import os
import unittest

from src.tokenizer import FoodTokenizer, basic_clean, whitespace_clean


class TestTokenizer(unittest.TestCase):
    def test_whitespace_clean(self):
        question = "   there    should    only be one    space between   words "
        ans = "there should only be one space between words"
        self.assertEqual(whitespace_clean(question), ans)

    def test_basic_clean(self):
        question = "&amp; The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."
        ans = "& The Mona Lisa doesn't have eyebrows."
        self.assertEqual(basic_clean(question), ans)

    def test_FoodTokenizer(self):

        # check configs and token paths exists
        configs_path = "src/model_configs/baseline.json"
        self.assertTrue(os.path.exists(configs_path))
        tokens_path = "src/model_configs/tokens_by_length.json"
        self.assertTrue(os.path.exists(tokens_path))

        # check tokenizer outputs a tensor of desired length
        with open(configs_path) as f:
            configs = json.load(f)
        tokenizer = FoodTokenizer(tokens_path, replace={"고깃": "고기"}, merge_strings=["김치"], configs=configs)
        test = tokenizer(["참치김치_찌개"])
        self.assertEqual(test.type(), "torch.LongTensor")
        self.assertTrue(0 not in test.tolist()[0])

        # test cache
        tokenizer(["김치", "김치"])

        # test tokenizer.v2
        self.assertEqual(tokenizer.v2("꽞"), [["꽞"]])

        # test how tokenizer deals with token of length 0
        temp = {"0": [""], "2": ["김치", "찌개"], "3": ["설렁_탕"]}
        token_path = "temp_tokens_by_length.json"
        with open(token_path, "w") as f:
            json.dump(temp, f)
        tokenizer = FoodTokenizer(token_path, configs=configs)
        os.remove((token_path))
        self.assertEqual(len(tokenizer.lengths), len(temp) - 1)
