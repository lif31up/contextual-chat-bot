import argparse

def train(path: str, save_to: str, iters: int) -> int:
  import src.train as trainer
  trainer.main(path.strip(), save_to, iters)
# init_train()

def eval(path: str) -> int:
  import src.eval as evaler
  evaler.main(path)
# init_eval()

def chat(path: str, response_path: str) -> int:
  import src.chat as chatter
  chatter.main(path, response_path)
# init_chat

def main():
  parser = argparse.ArgumentParser(description="maincmd")
  parser.add_argument("--path", type=str, help="path of your model")
  subparser = parser.add_subparsers(title="subcmd")

  # train
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="path to your model")
  parser_train.add_argument("--save-to", type=str, help="path to save your model")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train(kwargs.path, kwargs.save_to, kwargs.iters))

  # chat
  parser_chat = subparser.add_parser("chat", help="chat with your model")
  parser_chat.add_argument("--path", type=str, help="path to your model")
  parser_chat.add_argument("--response", type=str, help="path to your responses")
  parser_chat.set_defaults(func=lambda kwargs: chat(kwargs.path, kwargs.response))


  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  elif args.path: eval(args.path)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()