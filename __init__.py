import argparse

def main():
  parser = argparse.ArgumentParser(description="main cli")

  parser.add_argument("path", type=str, help="type your train set path")
  parser.add_argument("--iter", type=int, help="iteration numbers")
# main():




FAIL, SUCCESS = 1, 0
EXIT, NONE, TRAIN, EVAL, INSPECT = -1, 0, 1, 2, 3

def cmd_interface(string: str) -> int:
  buffer, ERROR = string.strip(), -2

  match buffer:
    case "exit": exit(buffer)
    case "train": return TRAIN
    case "eval": return EVAL
    case "inspect": return INSPECT
    case "": return NONE
  # match

  return ERROR
# cmd_interface

def init_train() -> int:
  import src.train as train
  file_path: str = input("* hint: press enter to use default trainset path\npath> ").strip()
  if file_path.strip() == "": file_path = "./src/data/raw/trainset.yml"
  try: return train.main(file_path)
  except Exception as e: print(f"error: {e}")
  return FAIL
# init_train()

def init_eval():
  import src.evaluate as eval
  eval.__main__()
# init_eval()

def init_inspect():
  print("not yet...")
# init_inspect

init_funcs = [lambda _=0: _, init_train, init_eval,init_inspect]

def main():
  epoch: int = 0
  while 1:
    if epoch > 100: print(f"prog] exit with maximum tries of {exit(-epoch)}")
    cmd_buffer = cmd_interface(input("command> "))
    if cmd_buffer == EXIT: exit(0)
    if init_funcs[cmd_buffer]() == FAIL: print(f"error: {init_funcs[cmd_buffer]}")
    epoch += 1
  # while
# __init__()
if __name__ == "__main__": main()