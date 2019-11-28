from pathlib import Path

DRY = False
root_path = Path("/datadisk/Data/IntegrationTests/ephys/choice_world_init")
root_path = Path("/mnt/s1/Data/Subjects/ZM_1735/2019-08-01/001/raw_ephys_data")
RENAMES = [('_spikeglx_ephysQcFreqAP.freq*.npy', '_iblqc_ephysSpectralDensityAP.freqs.npy'),
           ('_spikeglx_ephysQcFreqAP.power*.npy', '_iblqc_ephysSpectralDensityAP.power.npy'),
           ('_spikeglx_ephysQcFreqLF.freq*.npy', '_iblqc_ephysSpectralDensityLF.freqs.npy'),
           ('_spikeglx_ephysQcFreqLF.power*.npy', '_iblqc_ephysSpectralDensityLF.power.npy'),
           ('_spikeglx_ephysQcTimeAP.rms*.npy', '_iblqc_ephysTimeRmsAP.rms.npy'),
           ('_spikeglx_ephysQcTimeAP.times*.npy', '_iblqc_ephysTimeRmsAP.timestamps.npy'),
           ('_spikeglx_ephysQcTimeLF.rms*.npy', '_iblqc_ephysTimeRmsLF.rms.npy'),
           ('_spikeglx_ephysQcTimeLF.times*.npy', '_iblqc_ephysTimeRmsLF.timestamps.npy')]

for rename in RENAMES:
    for f2rename in root_path.rglob(rename[0]):
        print(f2rename, rename[1])
        if not DRY:
            f2rename.rename(f2rename.parent.joinpath(rename[1]))
