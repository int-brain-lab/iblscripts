file_mp4 =
overwrite =

def _format_timer(timer):
    """Formats timing information for the DLC pipeline"""
    logstr = ''
    for item in timer.items():
        logstr += f'\nTiming {item[0]}Camera [sec]\n'
        for subitem in item[1].items():
            logstr += f'{subitem[0]}: {int(np.round(subitem[1]))}\n'
    return logstr

timer = OrderedDict()
timer[f'{cam}'] = OrderedDict()
path_dlc = download_weights(version=version, one=self.one)

dlc_result, timer[f'{cam}'] = dlc(file_mp4, path_dlc=path_dlc, force=overwrite,
                                                          dlc_timer=timer[f'{cam}'])