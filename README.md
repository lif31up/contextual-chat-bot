`nltk` `torch` `yaml` `tqdm`
* **task**: classifying context of the input string, then the model responses based on it.
* **dataset**: integrated with `yaml` format document.

## Contextual Understanding Chatbot using Bag-of-Words (BoW)
This project is a Contextual Understanding Chatbot that uses Bag-of-Words (BoW) for text processing. The chatbot is designed to understand and respond to user inputs by converting text data into numerical representations, allowing the model to process and match patterns in conversations.

The chatbot leverages the Bag-of-Words (BoW) technique to represent user inputs as a collection of word frequency vectors. The model is trained to respond contextually based on pre-defined intents or keywords. This approach focuses on understanding the user's intent and matching it to appropriate responses.

[Test Result on Colab](https://colab.research.google.com/drive/1WGmmHb90CfPTgRTze4QkLJWD7cLWbkY8?usp=sharing)

### Data Preprocessing (BoW)
* Tokenization: Splitting the text into individual words.
* Lowercasing: Converting all text to lowercase for uniformity.
* Stop-word Removal: Removing common words (e.g., "the", "and", "is") that do not contribute to meaningful context.
* Stemming: Reducing words to their root form (e.g., "running" to "run").
* Bag-of-Words (BoW): Converting text into a fixed-length vector, where each element represents the frequency of a particular word from a vocabulary.

---
## Instruction
Organize your dataset into a structure compatible with PyTorch's ImageFolder:
```
dataset/
  ├── trainset.yml/
  │   ├── {tag, [responses]}
  │   ├── {tag, [responses]}
  │   └── ...
  └──
 ```

### Train Model
Train your model on a specified training dataset and set the number of iterations for training.
```
python run.py train --path ./to/your/dataset.yml --save-to ./save/to/here.pth --iters 1000
```
* `--path`: path to your training data file (yml).
* `--save_to`: path to save your model.
* `--iters`: number of training iterations to run. this controls how many times the model will learn from the data.

### Chat with Model
This command allows you to chat with the trained model. The chatbot will respond to your input based on its training.
```
python run.py --path ./to/your/model.path
```
* `--path`: Path to the trained model you wish to interact with.
---