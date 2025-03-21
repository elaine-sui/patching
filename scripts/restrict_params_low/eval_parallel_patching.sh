python src/parallel_patch.py   \
    --batch-size=128  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNetVal,MNISTVal,CarsVal,KITTIVal,SVHNVal,GTSRBVal  \
    --results-db=results_parallel_patch.jsonl  \
    --save=/pasteur/u/esui/patching/models/patch/ViTB32  \
    --data-location=/pasteur/u/esui/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --params_to_unfreeze=low \
    --wandb \
    --zeroshot_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_05-22_51_27/checkpoint_0.pt" \
    --mnist_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_05-22_51_27/checkpoint_5.pt" \
    --cars_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/CarsVal/2023_04_05-22_50_35/checkpoint_35.pt" \
    --kitti_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/KITTIVal/2023_04_05-22_51_39/checkpoint_40.pt" \
    --svhn_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/SVHNVal/2023_04_05-22_51_51/checkpoint_4.pt" \
    --gtsrb_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/GTSRBVal/2023_04_05-22_51_44/checkpoint_11.pt"