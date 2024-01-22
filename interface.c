#include <stdio.h>
#include <stdlib.h>

#define FAIL -1
int main(int argc, char *argv) {
  printf("This is discord bot runner. Hiana use deep learning speech recognition to play your favorite music\n");
  char *token = argv;
 token_cheker:
  if(*token == NULL){printf("token> "); scanf("%s", filename);}
  if(*token == NULL){goto token_checker;}

  char command[100]; sprintf(command, "python bot.py %s", token);
  error status = system(command);
  if(satatus == FAIL){printf("Failed to excute the command.\n"); goto token_checker;}

  return 0;
}  // main()
