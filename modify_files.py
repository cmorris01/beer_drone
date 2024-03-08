import os

def modify_class_label_to_one(directory):
    # Iterate through every file in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a text file
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Open the file and read lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Modify the first element (class label) of each line to 1
            modified_lines = ['1' + line[1:] if line.startswith('0') else line for line in lines]
            
            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(modified_lines)
            
            print(f"Modified {filename}")

# Specify the directory containing your YOLO formatted text files
directory = "/Users/coymorris/Desktop/beer_detector_dataset/beer_detector2023.v5i.yolov8/train/labels"
modify_class_label_to_one(directory)

print("Modification completed for all files.")
