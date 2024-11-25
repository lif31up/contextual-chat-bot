`nltk` `torch` `yaml` `tqdm`

## Contextual Understanding Chatbot using Bag-of-Words (BoW)
This project is a Contextual Understanding Chatbot that uses Bag-of-Words (BoW) for text processing. The chatbot is designed to understand and respond to user inputs by converting text data into numerical representations, allowing the model to process and match patterns in conversations.

The chatbot leverages the Bag-of-Words (BoW) technique to represent user inputs as a collection of word frequency vectors. The model is trained to respond contextually based on pre-defined intents or keywords. This approach focuses on understanding the user's intent and matching it to appropriate responses.

### Data Preprocessing (BoW)
* Tokenization: Splitting the text into individual words.
* Lowercasing: Converting all text to lowercase for uniformity.
* Stop-word Removal: Removing common words (e.g., "the", "and", "is") that do not contribute to meaningful context.
* Stemming: Reducing words to their root form (e.g., "running" to "run").
* Bag-of-Words (BoW): Converting text into a fixed-length vector, where each element represents the frequency of a particular word from a vocabulary.

## CLI
### 1. Evaluate Model
Use this command to evaluate your trained model on a specified dataset.
```
python run.py --path <path>
```
* `<path>`: Path to the model or dataset you want to evaluate.

### 2. Train Model
Train your model on a specified training dataset and set the number of iterations for training.
```
python run.py train --path <trainset_path> --iters <number_iterations>
```
* `<trainset_path>`: Path to your training data file (e.g., train.json or CSV).
* `<number_iterations>`: Number of training iterations to run. This controls how many times the model will learn from the data.

### 3. Chat with Model
This command allows you to chat with the trained model. The chatbot will respond to your input based on its training.
```
python run.py chat --path <model_path> --response <responses_path>
```
* `<model_path>`: Path to the trained model you wish to interact with.
* `<responses_path>`: Path to the responses file that contains predefined responses associated with various intents.
