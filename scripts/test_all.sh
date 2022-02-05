DATA=$1
GPU=$2
CKPT=$3
REFINE=$4 # Flag to perform keyframe finetuning
MULTISPRT=$5

SPLITS="0 1 2 3"
METHODS="repri tti"
SHOTS="1 5"
ARCHS="50"
VCWINS="[3,5,7,9,11]"
TESTNUM="10000"

for LAYERS in $ARCHS
do
    for SHOT in $SHOTS
    do
        if [ $SHOT == 1 ]
        then
           bsz_val="100"
        elif [ $SHOT == 5 ]
        then
           bsz_val="50"
        elif [ $SHOT == 10 ]
        then
           bsz_val="20"
        fi
        
        for METHOD in $METHODS
        do
            if [[ "$METHOD" == "ftune" ]]; then
                WEIGHTS="[1.0,0.0,0.0,0.0]"
            fi
            if [[ "$METHOD" == "repri" ]]; then
                WEIGHTS="[1.0,'auto','auto',0.0]"
            fi
            if [[ "$METHOD" == "tti" ]]; then
                WEIGHTS="[1.0,'auto','auto','auto']"
            fi

            for SPLIT in $SPLITS
            do
	            dirname="results/test/${METHOD}/arch=resnet-${LAYERS}/data=${DATA}/shot=shot_${SHOT}/ckpt=${CKPT}/refine=${REFINE}/multisprt=${MULTISPRT}/split=split_${SPLIT}"
                mkdir -p -- "$dirname"
                python3 -m src.test --config config_files/${DATA}.yaml \
                                    --opts train_split ${SPLIT} \
                                           batch_size_val ${bsz_val} \
                                           shot ${SHOT} \
                                           layers ${LAYERS} \
                                           FB_param_update "[10]" \
                                           temperature 20.0 \
                                           adapt_iter 50 \
                                           cls_lr 0.025 \
                                           gpus ${GPU} \
                                           test_num ${TESTNUM} \
                                           n_runs 5 \
                                           weights $WEIGHTS \
                                           workers 0 \
                                           vc_wins ${VCWINS} \
                                           ckpt_path ${CKPT} \
                                           refine_keyframes_ftune ${REFINE} \
                                           multi_rnd_sprt ${MULTISPRT} \
                                           | tee ${dirname}/log_${PI}.txt
            done
        done
    done
done
