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

def main():
  parser = argparse.ArgumentParser(description="maincmd")

  parser.add_argument("--path", type=str, help="path of your model")

  subparser = parser.add_subparsers(title="subcmd")
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="type your model path")
  parser_train.add_argument("--iters", type=int, help="type your iterations")
  parser_train.set_defaults(func=lambda path, iters: init_train(path, iters))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(path=args.path, iters=args.iters)
  elif args.path: init_eval(args.path)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()