python src/parallel_patch.py   \
    --batch-size=128  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNetVal,MNISTVal,CarsVal,KITTIVal,SVHNVal,GTSRBVal  \
    --results-db=results_parallel_patch.jsonl  \
    --save=/pasteur/u/esui/patching/models/patch/ViTB32  \
    --data-location=/pasteur/u/esui/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0