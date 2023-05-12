##
import numpy as np
from ibllib.atlas.regions import BrainRegions
import os
from one.api import ONE
from ibllib.atlas import atlas
from needles2.probe_model import ProbeModel
from pathlib import Path
from one.remote import aws
import pandas as pd

# Instantiate brain atlas and one
one = ONE()
ba = atlas.AllenAtlas(25)
ba.compute_surface()
br = BrainRegions()
# Remap volume from Allen to Beryl
label_beryl = ba.regions.mappings['Beryl'][ba.label]
# Get volume of acronyms
mapped_atlas_ac = br.acronym[label_beryl]  # TODO could be done faster by remapping list of brain reg to ID instead

# Saving paths
filepath_sp_vol = Path('/Users/gaelle/Desktop/Reports/Coverage/test/second_pass_volume.npy')
filepath_coverage = Path('/Users/gaelle/Desktop/Reports/Coverage/test/coverage.npy')
filepath_df_cov_val = filepath_coverage.parent.joinpath('df_cov_val.csv')

if not filepath_sp_vol.exists():
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file('resources/physcoverage/second_pass_volume.npy',
                         filepath_sp_vol, s3=s3, bucket_name=bucket_name)

##
'''
===========================
   BRAIN REGION COVERAGE
===========================
'''
# Overwrite according to Nick's selection
# https://docs.google.com/spreadsheets/d/1d6ghPpc2FT4D5t2n6eKYk8IcoOG4RgChaVLhazek04c/edit#gid=1168745718
acronyms_region_cov = ['AD', 'AHN', 'AUDpo', 'CL', 'COAa', 'FN', 'GPi', 'IO',
                       'PG', 'RE', 'STN', 'VTA']

##
'''
============================
    VOLUMETRIC COVERAGE
============================
'''
# Prepare SP volume
sp_volume = np.load(filepath_sp_vol)
sp_volume[sp_volume == 0] = np.nan
sp_volume = 2 * sp_volume
# Remove areas from this volume # 20th April 2023 decision - Phys WG
acronym_remove_sp = ['MOB', 'AOB', 'AOBgr', 'onl', 'AOBmi']
# === BRAIN REGION ===
for i_acronym in acronym_remove_sp:
    sp_volume[np.where(br.acronym[ba.label] == i_acronym)] = np.nan
    # Note : Use Allen parcellation instead of Beryl (mapped_atlas_ac) as not all voxels contained in aggregates
sp_voxels = np.where(~np.isnan(sp_volume.flatten()))[0]

# Init volume of points
breg_points = ba.label.copy()
breg_points[:] = 0
# For each brain region of interest, find voxel and priority, and assign point number
for i_acronym in acronyms_region_cov:
    breg_points[np.where(mapped_atlas_ac == i_acronym)] = 10

# Compute coverage (overwrite)
dist = 354
django_str = []
pr = ProbeModel(ba=ba, lazy=True)
pr.get_traj_for_provenance('Planned', django=django_str)
pr.get_traj_for_provenance('Micro-manipulator', django=django_str)
pr.get_traj_for_provenance('Histology track', django=django_str)
pr.get_traj_for_provenance('Ephys aligned histology track', django=django_str)
pr.get_traj_for_provenance(provenance='Ephys aligned histology track',
                           django=django_str + ['probe_insertion__json__extended_qc__'
                                  'alignment_resolved,True'], prov_dict='Resolved')
pr.find_traj_is_best(provenance='Planned')
pr.find_traj_is_best(provenance='Micro-manipulator')
pr.find_traj_is_best(provenance='Histology track')
pr.find_traj_is_best(provenance='Ephys aligned histology track')
pr.traj['Resolved']['is_best'] = np.arange(len(pr.traj['Resolved']['traj']))
pr.get_insertions_with_xyz()
pr.compute_best_for_provenance('Planned')

trajs = pr.traj['Best']['traj']
coverage, sp_per0, sp_per1, sp_per2 = pr.compute_coverage(trajs, dist_fcn=[dist, dist + 1], pl_voxels=sp_voxels)
df_coverage_vals = pd.DataFrame.from_dict({"0": sp_per0[-1], "1": sp_per1[-1], "2": sp_per2[-1]})

##
'''
============================
    COMBINE COVERAGES
============================

'''
# === VOLUMETRIC COVERAGE ===
cov_points = coverage.copy()
# Remove values outside SP mask and make sure all values for 2+ probes are 2
cov_points[np.isnan(sp_volume)] = np.nan
cov_points[np.where(cov_points > 2)] = 2
# Assign points
cov_points[np.where(cov_points == 0)] = 30
cov_points[np.where(cov_points == 1)] = 50
cov_points[np.where(cov_points == 2)] = 0
cov_points[np.isnan(cov_points)] = 0  # For summing

# === BRAIN REGION ===
# Init volume of points
breg_points = ba.label.copy()
breg_points[:] = 0
# For each brain region of interest, find voxel and priority, and assign point number
for i_acronym in acronyms_region_cov:
    breg_points[np.where(mapped_atlas_ac == i_acronym)] = 10

# === SUM the 2 matrices ===
sum_points = breg_points + cov_points
sum_points[np.where(sum_points == 0)] = np.nan

# Restrict to SP mask
sum_points[np.isnan(sp_volume)] = np.nan

# Save locally
np.save(filepath_coverage, sum_points)
df_coverage_vals.to_csv(filepath_df_cov_val)
# Synch to AWS
os.system(f"aws --profile ibl s3 cp {filepath_coverage} "
          f"s3://ibl-brain-wide-map-private/resources/physcoverage/{os.path.basename(filepath_coverage)}")
os.system(f"aws --profile ibl s3 cp {filepath_df_cov_val} "
          f"s3://ibl-brain-wide-map-private/resources/physcoverage/{os.path.basename(filepath_df_cov_val)}")
