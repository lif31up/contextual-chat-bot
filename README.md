# Contextual Understanding using torch ANN
this model does NLP with **bag of word** concept; its function is to sort any patterns into desirable context.

### Import
* `nltk`
* `torch`
* `yaml`
* `tqdm`

### Learn about its commands
this bot has evaluation as main-command and train as sub-command. Each of them has own arguments.

* `run --path <path>`: evaluate your model.
* `run train --path <trainset_path> --iters <number_iterations>`: train your model.
* `run chat --path <model_path> --response <responses_path>`: chat with your model.