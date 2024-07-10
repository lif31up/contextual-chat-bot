# Contextual Understanding using torch ANN
this model does NLP with **bag of word** concept; its function is to sort any patterns into desirable context.

### Import
* `nltk`
* `torch`
* `yaml`
* `tqdm`

### Learn about the commands
`python run.py --help`
* `python run --path <path>`: evaluate your model.
* `python run train --path <trainset_path> --iters <number_iterations>`: train your model.
* `python run chat --path <model_path> --response <responses_path>`: chat with your model.