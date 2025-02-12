import json
import os
import pandas as pd
from tqdm import tqdm

caption_path = '/mnt/sda1/saksham/TI2AV/AVSync15/cog_train_caption.json'
meta_path = '/home/sxk230060/TI2AV/misc/finetrainers/asva_scripts/AVSync15_metadata_valid.csv'
df = pd.read_csv(meta_path)
df = df[df['is_valid'] == True].reset_index(drop=True)
del df['is_valid']
df = df[df['split']=='train'].reset_index(drop=True)
caption = json.load(open(caption_path))

prompt_save_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/videos/prompts_filter.txt'
vid_save_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/videos/train_filter.txt'

for i, row in tqdm(df.iterrows(), total=len(df)):
    rel_path = os.path.join(row['label'], row['vid']+ '.mp4')
    with open(vid_save_path, 'a') as f:
        f.write(rel_path + '\n')
    
    with open(prompt_save_path, 'a') as f:
        f.write(caption[row['vid']].strip() + '\n')