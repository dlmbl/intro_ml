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