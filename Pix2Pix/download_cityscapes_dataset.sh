# taken and modified from https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh

FILE="cityscapes"
DATA_DIR='../data'

if [ ! -d $DATA_DIR ] 
then
    mkdir $DATA_DIR
fi

TARGET_DIR=${DATA_DIR}/${FILE}/
TAR_FILE=${DATA_DIR}/${FILE}.tar.gz

echo "Download $FILE to $OUTDIR"
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
wget -N $URL -O $TAR_FILE


mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C $DATA_DIR

echo "remove $TAR_FILE"
rm $TAR_FILE