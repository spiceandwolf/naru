python eval_model.py --dataset=power --glob='power-11.8MB-model29.329-data20.621-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-20epochs-seed0' --num-queries=2000 \
--residual --layers=5 --fc-hiddens=256 --direct-io --column-masking \
--input-encoding=binary --output-encoding=one_hot \
--run-sampling