from lzma import LZMACompressor


class Audio_data:
    def __init__(self,audio_signals = 0):
        

        self.__audio_signals = audio_signals

    def get_audio_signals(self):
        return self.__audio_signals
