"""
This is a script for creating a debug dataset of 20 fake images and 20 real images.
"""

import pandas as pd
import os
import torch
from torchvision.transforms.functional import to_tensor


def main():
    # get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assert current_dir.endswith("datasets")
    midjourney_data_dir = os.path.join(current_dir, "midjourney")
    # create the data directory if it doesn't exist
    image_size = 512  # size of the imagess
    resize_mode = "center_crop"
    disallowed_header_directives = (
        []
    )  # empty list means it ignores robot.txt and downloads anyway
    download_command = f"img2dataset --url_list {midjourney_data_dir} --input_format 'parquet'--url_col 'img_url' --caption_col 'text' --output_format webdataset   --output_folder {midjourney_data_dir} --processes_count 8 --thread_count 64 --image_size {image_size} --resize_only_if_bigger=True --resize_mode={resize_mode} --skip_reencode=True --disallowed_header_directives {disallowed_header_directives}"

    # TODO: download midjourney dataset to the cluster
    # TODO: Download the df = pd.read_csv("datasets/midjourney/midjourney_2022_reduced.csv")
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

    # TODO save this to a new location
    df_150.to_parquet(f'./datasets/midjourney/{data_dir}/seed_0.parquet')

    # TODO: run the img2dataset command wiht options at the top

    # TODO: read the file to verify things downloaded correctly
    # output_df = pd.read_parquet("datasets/midjourney/3k_subset_output/00000.parquet")

    #     for i in range(len(output_df)):
    # if i > 15:
    #     break
    # print(output_df["url"].iloc[i])

def clean_dataset():
    """
    This removes any images of the wrong size:
    """
    # Loop through each file in the directory
    directory = "./datasets/midjourney/3k_subset_output/unziped"

    n_of_size_1024 = 0
    n_of_size_512 = 0
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
            if x > 1024 or y > 1024:
            print("this shouldn't happen, image too big")
            elif x == 1024 and y == 1024:
            n_of_size_1024 += 1
            n_of_size_512 += 1
            midjourney_dataset.append(image_tensor)
            elif x >= 512 and y >= 512:
            n_of_size_512 += 1
            total += 1

            # # Print the shape of the tensor
            # print(filename, image_tensor.shape)
    print(f"n 1024: {n_of_size_1024}")
    print(n_of_size_512)
    print(total)


def save_to_folder():
    img_dataset = []

    for i in range(100):
    # row = df.iloc[i]
    # img_url = f"https://cdn.discordapp.com/attachments/{row.img_url}"
    # response = requests.get(img_url_1, stream=True)
    # img = Image.open(response.raw)
    img_tensor = to_tensor(img)
    img_dataset.append(img_tensor)
    if img_tensor.shape != torch.Size([3, 1664, 1664]):
        print(f"img actually has shape {img_tensor.shape}")
    stacked_img = torch.stack(midjourney_dataset)
    print(stacked_img.shape)
    torch.save(stacked_img, "datasets/midjourney/milestone_subset.pt") 



if __name__ == "__main__":
    main()
