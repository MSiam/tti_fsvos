## Pascal2MiniVSPW w/o + with Keyframe refinement
#bash scripts/test_all.sh inference/pascal2minivspw [1] model_ckpt/ False False
#bash scripts/test_all.sh inference/pascal2minivspw [1] model_ckpt/ True False
#
## Youtube-VIS V1 (Single Support set per class) w/o + with Keyframe refinement + DCL
#bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ False False False
#bash scripts/test.sh inference/ytvis 5 [0] 50 model_ckpt/ True False
#bash scripts/test.sh inference/ytvis 5 [0] 50 model_ckpt_dcl/ True False

#
## Youtube-VIS V2 (Multiple Support set per class) w/o + with Keyframe refinement
#bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ False False False

# MiniVSPW2MiniVSPW w/o + with Keyframe refinement for 1 and 5 shot
#bash scripts/test.sh inference/minivspw2minivspw 1 [1] 50 checkpoints_nminivspw/ False False
#bash scripts/test.sh inference/minivspw2minivspw 1 [0] 50 checkpoints_nminivspw/ True False
#bash scripts/test.sh inference/minivspw2minivspw 5 [1] 50 checkpoints_nminivspw/ False False
#bash scripts/test.sh inference/minivspw2minivspw 5 [0] 50 checkpoints_nminivspw/ True False

#echo "No Keyframe Refinement"
#bash scripts/test.sh inference/ytvis_vswin 5 [1] 50 /local/riemann1/home/msiam/checkpoints_nometalearning_videoswin/ False False False

#echo "With Keyframe Refinement"
bash scripts/test.sh inference/ytvis_vswin 5 [0] 50 /local/riemann1/home/msiam/checkpoints_nometalearning_videoswin/ True False False
