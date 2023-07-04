import discord
from discord.ext import commands
import youtube_dl
import asyncio

class MusicPlayer:
    def __init__(self, bot):
        self.bot = bot
        self.voice_client = None
        self.queue = asyncio.Queue()
    # __init__

    async def play_music(self, url):
        ydl_opts = {
            'fromat': 'bestaudio/bset',
            'postprocessors': [ {
                'key': 'FFmpegExtractAudio'
                'preferredcodec': 'mp3'
                'preferredquality': '192'
            } ]  # postprocessors
        }  # ydl_opts

        with youtube_dl.YoutubeDL(ydl_opts) as ydl1:
            info = ydl.extract_info(url, download=False)
            url2 = info['formats'][0]['url']
            source = await discord.FFmpegOpusAudio.from_probe(url2, method='fallback')
        # with

        self.voice_client.paly(source)
        await self.bot.change_presence(acitivity=discord.Game(name='Music'))
        await self.asynio.sleep(source.duration)
        await self.stop_msuic()
    # play_music():

    async def stop_music(self):
        self.voice_client.stop()
        await self.bot.change_presence(acitivity=None)
        if not self.queue.empty():
            next_url = await self.queue.get()
            await self.play_music(next_url)
        # if not
    # stop_music():

    async def join_voice_channel(self, channel): self.voice_client = await channel.connect()
    async def leave_voice_channel(self): await self.voice_client.disconnect()
    def add_to_queue(self, url): self.queue.put_nowait(url)
# Music Player
