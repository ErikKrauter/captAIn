import os
import shutil
import cv2

# first copy all images of faucets from image_source_dir to image_dest_dir
# then classify each image in image_dest_dir and store them in seperate subfolders
# then take one class as training class and retrieve the corresponding demonstrations from demo_source_dir and store
# those demos into demo_dest_dir

# Define the source and destination directories
image_source_dir = 'ManiSkill2/data/partnet_mobility/dataset/'  # here the original dataset is stored
# here I want to copy the images to and conduct the manual classifcation
image_dest_dir = 'ManiSkill2/data/partnet_mobility_filtered/Images/dataset_filtered/'

# this is the folder containing the class I want to train on, thus I need the corresponding demos
image_training_dir = 'ManiSkill2/data/partnet_mobility_filtered/Images/dataset_filterType2/1/Training'
# here all demos are located
demo_source_dir = 'ManiSkill2/demos/v0/rigid_body/TurnFaucet-v0/'
# here I copy the corresponding demos to. This is my training dataset
demo_dest_dir = 'ManiSkill2/demos/v0/rigid_body/TurnFaucet-v0_filteredTrainingDemos/'


# this function will copy trajectories from the demonstration dataset that belong to the images from png_folder
def copy_demonstrations(png_folder, demo_folder, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over all PNG files in the png_folder
    for file in os.listdir(png_folder):
        if file.endswith('.png'):
            # Extract the base name without the extension
            base_name = os.path.splitext(file)[0]

            # Find and copy corresponding h5 and json files from demo_folder
            for demo_file in os.listdir(demo_folder):
                if demo_file.startswith(base_name) and (demo_file.endswith('.h5') or demo_file.endswith('.json')):
                    source_path = os.path.join(demo_folder, demo_file)
                    destination_path = os.path.join(destination_folder, demo_file)
                    shutil.copy(source_path, destination_path)

# this function will display the images, wait for user to press key, create subfolder named after key and place
# image into that subfolder. In the end all images will be in subfolder for their corresponding class
# press q if image shall not be classified
def categorize_images(image_folder):
    # Create a list of image file paths
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_path in image_files:
        # Read and display the image
        img = cv2.imread(image_path)
        cv2.imshow('Faucet Image', img)
        key = cv2.waitKey(0)

        # Check if 'q' was pressed
        if key == ord('q'):
            print("Skipping image...")
        else:
            # Create a subfolder for the category if it doesn't exist
            category_folder = os.path.join(image_folder, chr(key))
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            # Move the image to the corresponding category folder
            destination_path = os.path.join(category_folder, os.path.basename(image_path))
            shutil.move(image_path, destination_path)

        # Close the image window
        cv2.destroyAllWindows()


def main(classifyImages, curateDemos):

    if classifyImages:
        # Create the destination directory if it doesn't exist
        # afterward iterate through the dataset of folder and move them to the destination
        if not os.path.exists(image_dest_dir):
            os.makedirs(image_dest_dir)

            # Iterate through each folder in the source directory
            for folder in os.listdir(image_source_dir):
                # Check if the folder name is a number between 5000 and 5076
                if folder.isdigit() and 5000 <= int(folder) <= 5076:
                    # Construct the path to the 'parts_render_after_merging' subfolder
                    subfolder_path = os.path.join(image_source_dir, folder, 'parts_render_after_merging')

                    # Check if the subfolder exists and contains the '0.png' file
                    file_path = os.path.join(subfolder_path, '0.png')
                    if os.path.exists(file_path):
                        # Construct the destination file path with the folder name
                        destination_file = os.path.join(image_dest_dir, f'{folder}.png')

                        # Copy and rename the file to the destination directory
                        shutil.copy(file_path, destination_file)
            # After the script is run, the destination directory will contain all the 0.png files renamed to their respective folder names
            print("All files copied successfully!")

        # if destination folder with all the images already exists, start manually classifying them
        else:
            # Run the categorization function
            categorize_images(image_dest_dir)
            print("Categorization complete.")
    if curateDemos:
        copy_demonstrations(image_training_dir, demo_source_dir, demo_dest_dir)


if __name__ == "__main__":
    classifyImages = False
    curateDemos = True

    main(classifyImages, curateDemos)
