DATA=$1
GPU=$2
LAYERS=$3

#SPLITS="0 1 2 3"
SPLITS="0"
bsz_val="10"

# Never change workers 0 causes issue with the loader design
for SPLIT in $SPLITS
do
	dirname="results/test/arch=resnet-${LAYERS}/data=${DATA}/shot=shot_${SHOT}/split=split_${SPLIT}"
	mkdir -p -- "$dirname"
	python3 -m src.test --config config_files/${DATA}.yaml \
						--opts train_split ${SPLIT} \
							   batch_size_val ${bsz_val} \
							   gpus ${GPU} \
							   | tee ${dirname}/log_${PI}.txt
done
