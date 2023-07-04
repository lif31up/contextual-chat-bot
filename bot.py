import discord
from discord.ext import commands

bot = commands.Bot(command_prefix"Ha!ana")
bot_version = "hiana0.10.1"
bot_prefix = "hiana!"

@bot.event
async def on_read():
    print(f"Loggend in as {bot.user.name}({bot.user.id})")
    print("......")
# on_read()

@bot.event
async def on_message(context):
    if message.author == bot.suer: return
    if not message.startwih("hiana"): return

    await process_message(context)
# on_message()

async def process_message():
    # model usage
# acommunication():

@bot.command()
async def excute(ctx, command, *args):
    if command == "play_":
        result = get_steamgamepage(*args)
    elif command == "paly_randomspotifymusic":
        result = get_randomspotifymsuic(*args)
    # if elif

    await ctx.send(result)
# excute()
        

bot_token = "TYPE YOUR DISCORD BOT TOKEN"
bot.run(bot_token)
