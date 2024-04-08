#!/bin/python3

#%%
from PIL import Image
import subprocess
import glob
import os

#%%
b_lam = 5.5*1.e-3
# b_lam = 9.*1.e-3

blstr = int(b_lam*1.e3) 
reppng = f"./bl_{blstr}/"

original_directory = os.getcwd()

w1 = 638
h1 = 639
#%% on retire les espaces des noms :
command_to_run = [f'getridofspacesinfilenames.sh > /dev/null 2>error.log']

os.chdir(reppng)

result = subprocess.run(command_to_run, shell=True, check=False)

os.chdir(original_directory)

#%%
all_files = glob.glob(reppng + "*.png")

#%%

images = [Image.open(f'{reppng}bl_{i}.png') for i,file in enumerate(all_files) ]
widths, heights = zip(*(i.size for i in images))

#%%
total_width = sum(widths)
max_height = max(heights)

# new_img = Image.new('RGB', (images[0].size[0], images[0].size[1]))
combined = images[0]

# x_offset = 0
for img in images[1:]:
    combined = Image.alpha_composite(combined,img)
    # new_img.paste(img, (0,0))
    # new_img.paste(img, (x_offset,0))
    # x_offset += img.width

# combined.resize((w1,h1))

combined.save(f'./combined_bl{blstr}.png')

# %%
images = [f"./combined_bl5_crop.png",f"./combined_bl9_crop.png"]
# bl = [5.5,9.]
for img in images:
  namesave = img.split(".png")[0]
  img2 = Image.open(img).resize((w1,h1))
  img2.save(f"{namesave}_resize.png")
# Image.open(f"{reppng}combined_bl5.png")
# Image.open(f"{reppng}combined_bl9.png")
# Image.open(f"{reppng}combined_bl9.png")

# %%
