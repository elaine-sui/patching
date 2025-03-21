for SEED in 1234 5678 910
do
    python src/sequential_patch.py   \
        --batch-size=128  \
        --model=ViT-B/32  \
        --eval-datasets=ImageNetVal,MNISTVal,CarsVal,KITTIVal,SVHNVal,GTSRBVal  \
        --results-db=results.jsonl  \
        --save=/pasteur/u/esui/patching/models/patch/ViTB32  \
        --data-location=/pasteur/u/esui/data \
        --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --seed $SEED \
        --params_to_unfreeze=last \
        --restrict_grad_dims \
        --k 50 \
        --wandb \
        --zeroshot_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_05-14_57_46/checkpoint_0.pt" \
        --mnist_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_05-14_57_46/checkpoint_5.pt" \
        --cars_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/CarsVal/2023_04_05-14_57_43/checkpoint_35.pt" \
        --kitti_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/KITTIVal/2023_04_05-14_57_35/checkpoint_40.pt" \
        --svhn_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/SVHNVal/2023_04_05-22_17_46/checkpoint_4.pt" \
        --gtsrb_ckpt "/pasteur/u/esui/patching/models/patch/ViTB32/GTSRBVal/2023_04_05-22_17_49/checkpoint_11.pt"
done