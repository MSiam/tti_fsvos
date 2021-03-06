DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4
SPLIT=$5

#SPLITS="0 1 2 3"
#SPLITS="0"

test_num=1000
run=test
if [ $DATA == "ytvis_episodic" ]
then
    run=test_ytvis
    test_num=0
fi

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

# Never change workers 0 causes issue with the loader design
dirname="results/test/arch=resnet-${LAYERS}/data=${DATA}/shot=shot_${SHOT}/split=split_${SPLIT}"
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
                           n_runs 5 \
                           weights "[1.0,'auto','auto','auto']"\
                           workers 0 \
                           | tee ${dirname}/log_${PI}.txt
