import discord
import mods

high_fn = {
    'music_search' : youtube.search,
    'music_recommandation' : youtube.sptf_chart,
    'play_playlist' : youtube.playlist
}  # interface

def cmd(tag):
    if tag in high_fn:
        print("log> user used high level function")
        global intents

        await message.channel.send(intents[])
        keyword = blabla
        await message.channel.send( fn[tag](keyword) )

    if tag in low_fn:
        print("log> user used low level function")
        await message.channel.send( fn[tag]() )
# cmd():
