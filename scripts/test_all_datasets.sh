# Pascal2MiniVSPW w/o + with Keyframe refinement
bash scripts/test_all.sh inference/pascal2minivspw [1] model_ckpt/ False False
bash scripts/test_all.sh inference/pascal2minivspw [1] model_ckpt/ True False

# Youtube-VIS V1 (Single Support set per class) w/o + with Keyframe refinement + DCL
bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ False False
bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ True False
bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt_dcl/ True False


# Youtube-VIS V2 (Multiple Support set per class) w/o + with Keyframe refinement
bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ False True
bash scripts/test.sh inference/ytvis 5 [1] 50 model_ckpt/ True True

# MiniVSPW2MiniVSPW w/o + with Keyframe refinement for 1 and 5 shot
bash scripts/test.sh inference/minivspw2minivspw 1 [1] 50 checkpoints_nminivspw/ False False
bash scripts/test.sh inference/minivspw2minivspw 1 [1] 50 checkpoints_nminivspw/ True False

bash scripts/test.sh inference/minivspw2minivspw 5 [1] 50 checkpoints_nminivspw/ False False
bash scripts/test.sh inference/minivspw2minivspw 5 [1] 50 checkpoints_nminivspw/ True False
