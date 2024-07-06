python create_pretraining_data.py \
  --input_file=../train_scripts/renal_data.txt \
  --output_file=./renal_data.tfrecord \
  --vocab_file=../Models/biobert_v1.1_pubmed/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=300 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python run_pretraining.py \
  --input_file=./renal_data.tfrecord \
  --output_dir=../Models/ckg_biobert\
  --do_train=True \
  --do_eval=True \
  --bert_config_file=../Models/biobert_v1.1_pubmed/bert_config.json \
  --init_checkpoint=../Models/biobert_v1.1_pubmed/model.ckpt-1000000 \
  --train_batch_size=256 \
  --max_seq_length=300 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5