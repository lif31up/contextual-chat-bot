import discord
from discord.ext import commands

async def voice_update(member, before, after):
    if member == client.user:
        if before.channel is None and after.channel is not None:
            audio_recorder = await after.channel.connect()
            while audio_recorder.is_connected():
                audio_data = await audio_recorder.receive()
                audio_segment = AudioSegment(
                    data = audio_data[1], sample_width = 2, frame_rate = 48000, channels = 2
                )  # audio_segment
                audio_segment.export("recorded_speech.mp3", format="mp3")

                if audio_data[1] == b'':
                    await audio_recorder.disconnect()
                    break
                # if
            # while
# voice_update():
