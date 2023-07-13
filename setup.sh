# Create mamba environment with ipykernel for jupyter
mamba create --name 01_intro_ml python=3.10 \
    matplotlib numpy imageio scikit-learn scikit-image \
    tensorflow tensorflow_addons pandas tqdm ipykernel

# Activate base just in case
mamba activate base

# Create data directory
mkdir data
mkdir data/zips
cd data/zips

# Download and unzip the data
wget https://data.broadinstitute.org/bbbc/BBBC048/BBBC048v1.zip
unzip BBBC048v1.zip

# Unzip internal files
unzip CellCycle.zip -d ../

cd ../..