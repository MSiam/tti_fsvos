DATA=$1
SPLIT=$2
GPU=$3
LAYERS=$4


#SPLITS="0 1 2 3"
SPLITS="0"

#for SPLIT in $SPLITS
#do
dirname="results/train/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"
python3 -m src.train --config config_files/${DATA}.yaml \
                     --opts train_split ${SPLIT} \
                            layers ${LAYERS} \
                            gpus ${GPU} \
                            visdom_port 28333 \
                            visdom_env ${DATA}_${SPLIT} \
                            debug True\
                             | tee ${dirname}/log.txt
#done
