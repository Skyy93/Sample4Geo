from yacs.config import CfgNode as CN
import os

cfg = CN()

cfg.DataRoot = 'data'
cfg.data_path = cfg.DataRoot

#####################################################################################################

cfg.train_csv = os.path.join(cfg.data_path,'train_df.csv')
cfg.validate_csv = os.path.join(cfg.data_path,'validate_df.csv')
cfg.test_csv = os.path.join(cfg.data_path,'test_df.csv')

####################################spectrum4geo:used parameter#############################################################
cfg.sat_audio_path = os.path.join(cfg.data_path,'raw_audio')
cfg.sat_audio_tensors_path = os.path.join(cfg.data_path,'raw_audio_tensors')
cfg.sat_audio_spectrograms_path = os.path.join(cfg.data_path,'spectrograms')

cfg.sat_image_path = os.path.join(cfg.data_path,'images')