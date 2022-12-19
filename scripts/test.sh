DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4
CKPT=$5
REFINE=$6 # Flag to perform keyframe finetuning
MULTISPRT=$7 # Flag specific to ytvis to enable multiple support per class
ADAPKSHOT=$8 # Flag to specify adaptive kshot

NRUNS=1
SPLITS="0"
VCWINS="[3,5,7,9,11]"

run=test_nonbatched
test_num=0

################ bsz_val only used in Pascal-to-MiniVSPW for batching and faster inference
if [ $SHOT == 1 ]
then
   bsz_val="100"
elif [ $SHOT == 5 ]
then
   bsz_val="100"
elif [ $SHOT == 10 ]
then
   bsz_val="50"
fi

################# "selected_weights": Variable used only during quick validation
################# Never change workers 0 causes issue with the loader design
for SPLIT in $SPLITS
do
	dirname="results/test/arch=resnet-${LAYERS}/data=${DATA}/shot=shot_${SHOT}/ckpt=${CKPT}/refine=${REFINE}/multisprt=${MULTISPRT}/split=split_${SPLIT}"
	mkdir -p -- "$dirname"
	python3 -m src.$run --config config_files/${DATA}.yaml \
						--opts train_split ${SPLIT} \
							   batch_size_val ${bsz_val} \
							   shot ${SHOT} \
							   layers ${LAYERS} \
							   FB_param_update "[10]" \
							   temperature 20.0 \
							   adapt_iter 50 \
							   cls_lr 0.025 \
							   gpus ${GPU} \
							   test_num $test_num \
							   n_runs $NRUNS \
                               weights "[1.0,'auto','auto','auto']"\
                               workers 0 \
                               vc_wins ${VCWINS} \
                               ckpt_path ${CKPT} \
                               refine_keyframes_ftune ${REFINE} \
                               selected_weights "[1.0, 'auto', 'auto', 0.0]" \
                               multi_rnd_sprt ${MULTISPRT} \
                               adap_kshot ${ADAPKSHOT} \
							   | tee ${dirname}/log_${PI}.txt
done
                               #selected_weights [] \
