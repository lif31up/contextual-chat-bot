# Contextual Understanding using torch ANN
this model does NLP with **bag of word** concept; its function is to sort any patterns into desirable context.

### Import
* `nltk`
* `torch`
* `yaml`
* `tqdm`

### Learn about its commands
this bot has evaluation as main-command and train as sub-command. Each of them has own arguments.

* `--path`: path of your pth file.
* `train`
  * `--path <trainset_path>`: path of your train set file(yml format)
  * `--iters <number_iterations>`: numbers to your model iterate for training