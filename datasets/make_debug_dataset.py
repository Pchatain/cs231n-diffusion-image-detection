"""
This is a script for creating a debug dataset of 20 fake images and 20 real images.

To download the midjourney dataset, run the following command:

"""

import pandas as pd
import os
import img2dataset
import numpy as np
import torch
# import subprocess
import tarfile
import glob
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor


def main():
    DATASET_NAME = "debug_dataset"
    # get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assert current_dir.endswith("datasets")
    midjourney_data_dir = os.path.join(current_dir, "midjourney")
    output_dataset_dir = os.path.join(midjourney_data_dir, DATASET_NAME)

    df = pd.read_csv(os.path.join(midjourney_data_dir, "midjourney_2022_reduced.csv"))
    np.random.seed(0)
    df_150 = df.sample(n=20)
    print(len(df_150))

    # Verify if things look right
    # reformat the dataset to actually have urls
    # Define the prefix to append to each image URL
    prefix = "https://media.discordapp.net/attachments/"
    # Iterate over the rows of the dataframe and update the 'img_url' column
    for index, row in df_150.iterrows():
        df_150.at[index, 'img_url'] = f"{prefix}{row['img_url']}"
    # Print the first url to verify it's viewable
    for i in range(10):
        print(df_150['img_url'].iloc[i])

    df_150.to_parquet(f'{output_dataset_dir}/seed_0_{DATASET_NAME}.parquet')

    # create the data directory if it doesn't exist
    image_size = 512  # size of the imagess
    resize_mode = "center_crop"
    disallowed_header_directives = (
        []
    )  # empty list means it ignores robot.txt and downloads anyway
    # download_command = f"img2dataset --url_list {midjourney_data_dir} --input_format 'parquet'--url_col 'img_url' --caption_col 'text' --output_format webdataset   --output_folder {midjourney_data_dir} --processes_count 8 --thread_count 64 --image_size {image_size} --resize_only_if_bigger=True --resize_mode={resize_mode} --skip_reencode=True --disallowed_header_directives {disallowed_header_directives}"
    input_format = 'parquet'
    url_col = 'img_url'
    caption_col = 'text'
    output_format = 'webdataset'
    processes_count = 8
    thread_count = 64
    resize_only_if_bigger = True
    skip_reencode = True

    # Call the download function
    # fails if the files have already been downloaded, so we check if they exist first
    if not os.path.exists(os.path.join(output_dataset_dir, "00000.parquet")):
        print("Downloading dataset")
        img2dataset.download(
            url_list=output_dataset_dir,
            input_format=input_format,
            url_col=url_col,
            caption_col=caption_col,
            output_format=output_format,
            output_folder=output_dataset_dir,
            processes_count=processes_count,
            thread_count=thread_count,
            image_size=image_size,
            resize_only_if_bigger=resize_only_if_bigger,
            resize_mode=resize_mode,
            skip_reencode=skip_reencode,
            disallowed_header_directives=disallowed_header_directives
        )

    # TODO: read the file to verify things downloaded correctly
    output_df = pd.read_parquet(os.path.join(output_dataset_dir, "00000.parquet"))

    for i in range(len(output_df)):
        if i > 15:
            break
        print(output_df["url"].iloc[i])

    # extract all files
    # check if a .jpg file extension exists, and if so don't extract
    if len(glob.glob(os.path.join(output_dataset_dir, "*.jpg"))) == 0:
        print("Extracting tar files")
        # get all files ending in .tar
        tar_files = [f for f in os.listdir(output_dataset_dir) if f.endswith(".tar")]
        for tar_file in tar_files:
            tar_file = os.path.join(output_dataset_dir, tar_file)
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(output_dataset_dir)

    tensor_dataset_path = os.path.join(output_dataset_dir, 'milestone_subset.pt')
    if not os.path.exists(tensor_dataset_path):
        print("Creating tensor dataset")
        cleaned_images = clean_dataset(output_dataset_dir, image_size)
        cleaned_images_stacked = stack_dataset(cleaned_images, image_size)
        torch.save(cleaned_images_stacked, tensor_dataset_path) 



def clean_dataset(directory: str, target_size: int = 512):
    """
    This removes any images of the wrong size:
    """
    # Loop through each file in the directory
    # directory = "./datasets/midjourney/3k_subset_output/unziped"
    print(f"Looking for files in {directory}")

    n_of_correct_size = 0
    total = 0
    midjourney_dataset = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # Load the image using Pillow
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)

            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Convert the NumPy array to a PyTorch tensor
            image_tensor = torch.from_numpy(image_array)

            # Process the image tensor as needed
            x, y, c = image_tensor.shape
            if x > target_size or y > target_size:
                print("this shouldn't happen, image too big")
            elif x == target_size and y == target_size:
                n_of_correct_size += 1
                midjourney_dataset.append(image_tensor)
            total += 1

            # # Print the shape of the tensor
            # print(filename, image_tensor.shape)
    print(f"Found {n_of_correct_size} images of the right size {target_size}")
    print(f"percent of images of the right size {target_size}: {100 * n_of_correct_size/total}")
    return midjourney_dataset


def stack_dataset(cleaned_images, target_size: int = 512):
    """
    Takes immage tensors of shape (target_size, target_size, 3) and stacks them into a single tensor of shape (n_images, target_size, target_size, 3)
    3 is the number of color channels
    """
    img_dataset = []
    for img in cleaned_images:
        # img_tensor = to_tensor(img)
        assert img.shape == torch.Size([target_size, target_size, 3]), f"img actually has shape {img_tensor.shape} when it should have shape {torch.Size([3, target_size, target_size])}"
        img_dataset.append(img)
    stacked_img = torch.stack(img_dataset)
    print(stacked_img.shape)
    return stacked_img



if __name__ == "__main__":
    main()
