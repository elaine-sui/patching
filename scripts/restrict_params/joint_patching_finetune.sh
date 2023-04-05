python src/patch.py   \
    --train-dataset=CarsVal,GTSRBVal,KITTIVal,MNISTVal,SVHNVal  \
    --eval-every-epoch \
    --epochs=40  \
    --lr=0.001  \
    --warmup_length 200 \
    --batch-size=128  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNetVal,CarsVal,GTSRBVal,KITTIVal,MNISTVal,SVHNVal  \
    --results-db=results.jsonl  \
    --save=/pasteur/u/esui/patching/models/patch/ViTB32  \
    --data-location=/pasteur/u/esui/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --params_to_unfreeze=last \
    --wandb