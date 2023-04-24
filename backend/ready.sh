#!/bin/bash

# Starting in repo "backend"
echo "Starting in the 'backend' repo."

# Going a level above
cd ../../
echo "Moved a level above to $(pwd)"

# Cloning the git repository methods2test
git clone https://github.com/microsoft/methods2test.git
echo "Cloned the 'methods2test' repository."

# Git LFS pull
cd methods2test
git lfs pull
echo "Performed 'git lfs pull'."

# Going back into 'backend'
cd ../project/backend
echo "Moved back into the 'backend' repo."

# Unarchiving train.tar.bz2, eval.tar.bz2, and test.tar.bz2
tar -xjf ../../methods2test/corpus/raw/fm_fc_co/train.tar.bz2 -C ../../methods2test/corpus/raw/fm_fc_co
tar -xjf ../../methods2test/corpus/raw/fm_fc_co/eval.tar.bz2 -C ../../methods2test/corpus/raw/fm_fc_co
tar -xjf ../../methods2test/corpus/raw/fm_fc_co/test.tar.bz2 -C ../../methods2test/corpus/raw/fm_fc_co

echo "Unarchived train.tar.bz2, eval.tar.bz2, and test.tar.bz2."

# Copying the resulting train, eval, and test folders to corpus/raw/
cp -r ../../methods2test/corpus/raw/fm_fc_co/train ../data/raw/
cp -r ../../methods2test/corpus/raw/fm_fc_co/eval ../data/raw/
cp -r ../../methods2test/corpus/raw/fm_fc_co/test ../data/raw/
echo "Copied the train, eval, and test folders to ../data/raw/."

echo "Script completed successfully."