# Tracking script by S. Bourdin
# assumes that the first argument is the path to the input file 
# e.g., model_outputs/AIFS_init-2023.01.01-00h00_max-lead-120.nc

track () {
# Define file names
IN_FILE=$1
TMP_FILE="tmp/${IN_FILE:5:100}"
NODE_FILE="nodes/${IN_FILE:5:-3}"
TRACKS_FILE="tracks/${IN_FILE:5:-3}.csv"

if ! [ -f $TRACKS_FILE ] #Don't run if the tracks file already exists
then

# Pre-processing input file
# make sure that the model outputs include the following fields:
# - msl (mean sea level pressure)
# - u10 (10m zonal wind)
# - v10 (10m meridional wind)
# - z300 (300hPa geopotential height)
# - z500 (500hPa geopotential height)
# Make sure that the individual files have a valid_time axis, using a pd.DatetimeIndex
# Assuming your preprocessing script is a python script called preprocess.py, you could run it as follows:
python preprocess.py $IN_FILE

# DetectNodes
/tempestextremes_dir/bin/DetectNodes \
--in_data $TMP_FILE \
--out $NODE_FILE \
--searchbymin "msl" \
--closedcontourcmd "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0" \
--mergedist 6.0 \
--outputcmd  "msl,min,0;_VECMAG(u10,v10),max,2"

# StitchNodes
/tempestextremes_dir/bin/StitchNodes \
        --in $NODE_FILE \
        --out $TRACKS_FILE \
        --in_fmt "lon,lat,slp,wind10" \
        --range 8.0 \
        --mintime "12h" \
        --threshold "wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1" \
        --out_file_format "csv"

rm $TMP_FILE

else
        echo Already tracked $f
fi
}

FLIST=`ls data/*`
for f in $FLIST
do
        track $f
done

