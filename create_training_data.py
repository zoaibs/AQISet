import os
import random
import shutil

source = 'test_validate'
dest = 'validate_data'
files = os.listdir(source)
no_of_files = 30 #70-30 training testing split

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)