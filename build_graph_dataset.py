# generate dataset
from utils import build_dataset
args={}
args['input_csv'] = 'data/AquaTox.csv'
args['output_bin'] = 'data/AquaTox.bin'
args['output_csv'] = 'data/AquaTox_group.csv'

build_dataset.built_data_and_save_for_splited(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )





