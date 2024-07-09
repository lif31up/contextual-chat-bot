import argparse

FAIL, SUCCESS = 1, 0

def init_train(path: str, iters: int) -> int:
  import src.train as train
  error_code = FAIL

  try:
    train.main(path.strip(), iters)
    error_code = SUCCESS
  except Exception as e: print(f"{e}")

  return error_code
# init_train()

def init_eval(path: str) -> int:
  import src.eval as eval
  eval.main(path)
# init_eval()

def init_chat(model_path: str, responses_path: str) -> int:
  import src.chat as chat
  chat.main(model_path, responses_path)
# init_chat

def main():
  parser = argparse.ArgumentParser(description="maincmd")

  parser.add_argument("--path", type=str, help="path of your model")

  subparser = parser.add_subparsers(title="subcmd")

  # train
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="path to your model")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for training")
  parser_train.set_defaults(func=lambda kwargs: init_train(kwargs.path, kwargs.iters))

  # chat
  parser_chat = subparser.add_parser("chat", help="chat with your model")
  parser_chat.add_argument("--path", type=str, help="path to your model")
  parser_chat.add_argument("--response", type=str, help="path to your responses")
  parser_chat.set_defaults(func=lambda kwargs: init_chat(kwargs.path, kwargs.response))


  args = parser.parse_args()
  print(args)
  if hasattr(args, 'func'): args.func(args)
  elif args.path: init_eval(args.path)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()