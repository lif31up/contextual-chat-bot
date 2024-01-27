import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
from src.data.tokenize import tokenize, bag_of_words


class YamlLoader(Dataset):
    def __init__(self, filename):
        self.tags, self.dictionary = list(), list()

        with open(filename) as y_file:
            self.data = yaml.load(y_file, Loader=yaml.FullLoader)

        xy = list()
        for intent in self.data:
            tag = intent["tag"]
            self.tags.append(tag)
            for pattern in intent["patterns"]:
                tokenized_pattern = tokenize(pattern)
                xy.append([tokenized_pattern, tag])
                self.dictionary.extend(tokenized_pattern)
        # for for
        self.dictionary = sorted(self.dictionary)
        self.tags = sorted(self.tags)

        patterns, targets = list(), list()
        for x, y in xy:
            bag = bag_of_words(x, self.dictionary)
            patterns.append(bag)
            targets.append(self.tags.index(y))
        # for

        self.patterns = np.array(patterns)
        self.targets = (torch.from_numpy(np.array(targets))).type(torch.long)
        del xy, patterns, targets

        self.n_patterns = len(self.patterns)
        self.n_tags = len(self.tags)
        self.data_size = len(self.patterns[0])

    # __init__():

    def __getitem__(self, index):
        return self.patterns[index], self.targets[index]

    def __len__(self):
        return self.n_patterns


# yaml_loader
