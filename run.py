import argparse
from src import train
from src import chat

def main():
  parser = argparse.ArgumentParser(description="maincmd")
  parser.add_argument("--path", type=str, help="path of your model")
  subparser = parser.add_subparsers(title="subcmd")

  # train
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="path to your model")
  parser_train.add_argument("--save-to", type=str, help="path to save your model")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train.main(path=kwargs.path, save_to=kwargs.save_to, iters=kwargs.iters))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  elif args.path: chat.main(path=args.path)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()