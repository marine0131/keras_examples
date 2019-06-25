import os
import random
import shutil

def copyFile(src_dir, dst_dir):
    src_files = os.listdir(src_dir)

    sample = random.sample(src_files, 1000)

    for name in sample:
        shutil.copyfile(os.path.join(src_dir,name), os.path.join(dst_dir, name))


if __name__ == "__main__":
    src_dir = "../../training_data/cigarette"
    dst_dir = "./cigarette"
    copyFile(src_dir, dst_dir)
