#!/usr/bin/python3
#%%
import os
from PIL import Image
#%%
def create_gif_from_png_files(
    input_directory, output_directory, output_file, duration=6000,
):
    # Get all .png files in the input directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Sort the files numerically
    # png_files.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    sorted_files = sorted(
        png_files, key=lambda x: int(
            ''.join(filter(str.isdigit, x)),
        ),
    )

    # Open all images
    images = [
        Image.open(os.path.join(input_directory, f))
        for f in sorted_files
    ]

    # Save as GIF
    images[0].save(
        os.path.join(output_directory, output_file),
        save_all=True,
        append_images=images[1:],
        duration=duration // len(images),  # Calculate duration per frame
        loop=3,  # Loop indefinitely
    )

input_dir = './bl_5/'
output_file = 'bl_5.gif'
output_dir_gifs = 'gifs'
if not os.path.exists(output_dir_gifs):
    os.makedirs(output_dir_gifs)

create_gif_from_png_files(input_dir, output_dir_gifs, output_file)

# %%
