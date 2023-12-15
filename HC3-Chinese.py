# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\

"""# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Human ChatGPT Comparison Corpus (HC3) Chinese Version
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/Hello-SimpleAI/chatgpt-comparison-detection"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_BASE_URL = 'https://www.modelscope.cn/api/v1/datasets/simpleai/HC3-Chinese/repo?Revision=master&FilePath='
# _URLS = {
#     "all":"https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/all.jsonl",
#     "baike": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/baike.jsonl",
#     "open_qa": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/open_questions.jsonl",
#     "nlpcc_dbqa": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/nlpcc_dbqa.jsonl",
#     "finance": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/finance.jsonl",
#     "medicine": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/medicine.jsonl",
#     "law": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/law.jsonl",
#     "psychology": "https://huggingface.co/datasets/Hello-SimpleAI/chatgpt-comparison-corpus-chinese/resolve/main/zh/psychology.jsonl",
# }




# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """Human ChatGPT Comparison Corpus (HC3) Chinese Version"""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="all", version=VERSION, description="concatenation of all other subsets"),
        datasets.BuilderConfig(name="baike", version=VERSION, description="baike"),
        datasets.BuilderConfig(name="open_qa", version=VERSION, description="open_questions"),
        datasets.BuilderConfig(name="nlpcc_dbqa", version=VERSION, description="nlpcc_dbqa"),
        datasets.BuilderConfig(name="finance", version=VERSION, description="finance"),
        datasets.BuilderConfig(name="medicine", version=VERSION, description="medicine"),
        datasets.BuilderConfig(name="law", version=VERSION, description="law"),
        datasets.BuilderConfig(name="psychology", version=VERSION, description="psychology"),
    ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "all":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "id":datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "human_answers": datasets.Sequence(datasets.Value("string")),
                    "chatgpt_answers": datasets.Sequence(datasets.Value("string")),
                    "source": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            )
        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "id":datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "human_answers": datasets.Sequence(datasets.Value("string")),
                    "chatgpt_answers": datasets.Sequence(datasets.Value("string")),
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        # urls = _URLS[self.config.name]
        urls = _BASE_URL+self.config.name+'.jsonl'
        data_path = dl_manager.download_and_extract(urls) # for online datasets
        # data_path = _URLS[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    # "filepath": os.path.join(data_dir, "train.jsonl"),
                    "filepath": data_path,
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "test.jsonl"),
            #         "split": "test"
            #     },
            # ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "all":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "id": key,
                        "question": data["question"],
                        "human_answers": data["human_answers"],
                        "chatgpt_answers": data["chatgpt_answers"],
                        "source": data["source"]
                    }
                else:
                    yield key, {
                        "id": key,
                        "question": data["question"],
                        "human_answers": data["human_answers"],
                        "chatgpt_answers": data["chatgpt_answers"],
                    }
