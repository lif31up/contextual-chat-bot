`nltk` `torch` `yaml` `tqdm`
* **task**: classifying context of the input string, then the model responses based on it.
* **dataset**: integrated with `yaml` format document.

## Contextual Understanding Chatbot using Bag-of-Words (BoW)
This project is a Contextual Understanding Chatbot that uses Bag-of-Words (BoW) for text processing. The chatbot is designed to understand and respond to user inputs by converting text data into numerical representations, allowing the model to process and match patterns in conversations.

The chatbot leverages the Bag-of-Words (BoW) technique to represent user inputs as a collection of word frequency vectors. The model is trained to respond contextually based on pre-defined intents or keywords. This approach focuses on understanding the user's intent and matching it to appropriate responses.

### Data Preprocessing (BoW)
* Tokenization: Splitting the text into individual words.
* Lowercasing: Converting all text to lowercase for uniformity.
* Stop-word Removal: Removing common words (e.g., "the", "and", "is") that do not contribute to meaningful context.
* Stemming: Reducing words to their root form (e.g., "running" to "run").
* Bag-of-Words (BoW): Converting text into a fixed-length vector, where each element represents the frequency of a particular word from a vocabulary.

---
## Instruction
### Evaluate Model
Use this command to evaluate your trained model on a specified dataset.
```
python run.py --path <path>
```
* `<path>`: Path to the model or dataset you want to evaluate.

### Train Model
Train your model on a specified training dataset and set the number of iterations for training.
```
python run.py train --path <trainset_path> --save-to <model_path> --iters <number_iterations>
```
* `<trainset_path>`: Path to your training data file (e.g., train.json or CSV).
* `<number_iterations>`: Number of training iterations to run. This controls how many times the model will learn from the data.

### Chat with Model
This command allows you to chat with the trained model. The chatbot will respond to your input based on its training.
```
python run.py chat --path <model_path> --response <responses_path>
```
* `<model_path>`: Path to the trained model you wish to interact with.
* `<responses_path>`: Path to the responses file that contains predefined responses associated with various intents.
---
### 모델 평가
훈련된 모델을 특정 데이터셋에서 평가하려면 아래 명령어를 사용하세요.
```
python run.py --path <path>
```
* `<path>`: 평가하려는 모델 또는 데이터셋의 경로를 지정합니다.
### 모델 훈련
지정된 훈련 데이터셋을 기반으로 모델을 학습시키고, 학습 반복 횟수를 설정합니다.
```
python run.py chat --path <model_path> --response <responses_path>
```
* `<trainset_path>`: 훈련 데이터 파일의 경로 (예: `train.json`, `train.csv`).
* `<model_path>`: 학습된 모델을 저장할 경로를 지정합니다.
* `<number_iterations>`: 학습 반복 횟수. 데이터에서 학습을 수행하는 횟수를 설정합니다.
### 대화하기
훈련된 모델과 대화를 나눌 수 있습니다. 챗봇은 훈련 데이터를 기반으로 사용자의 입력에 응답합니다.
```
python run.py chat --path <model_path> --response <responses_path>
```
* `<model_path>`: 상호작용할 훈련된 모델의 경로를 지정합니다.
* `<responses_path>`: 다양한 의도(intent)에 대한 사전 정의된 응답을 포함한 파일 경로를 지정합니다.
