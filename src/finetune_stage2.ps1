# ---------------- setup ----------------
$env:CUDA_VISIBLE_DEVICES="0"

$model_name_or_model_dir = "stage1_outputs"

$train_data = "data\train_noise.jsonl"
$val_data   = "data\val_noise.jsonl"  

$output_dir = "stage2_outputs"
$log_file   = "$output_dir\log.txt"
if (-Not (Test-Path $output_dir)) { New-Item -ItemType Directory -Path $output_dir }

# ---------------- train ----------------
python FunASR-main\funasr\bin\train_ds.py `
++model=$model_name_or_model_dir `
++train_data_set_list=$train_data `
++valid_data_set_list=$val_data `
++dataset="AudioDatasetHotword" `
++dataset_conf.index_ds="IndexDSJsonl" `
++dataset_conf.batch_size=6000 `
++dataset_conf.sort_size=1024 `
++dataset_conf.batch_type="token" `
++dataset_conf.num_workers=4 `
++train_conf.max_epoch=50 `
++train_conf.log_interval=1 `
++train_conf.resume=true `
++train_conf.validate_interval=2000 `
++train_conf.save_checkpoint_interval=2000 `
++train_conf.avg_keep_nbest_models_type='loss' `
++train_conf.keep_nbest_models=20 `
++train_conf.avg_nbest_model=10 `
++train_conf.use_deepspeed=false `
++train_conf.find_unused_parameters=true `
++optim_conf.lr=0.0002 `
++output_dir=$output_dir 2>&1 | Tee-Object -FilePath $log_file

Write-Host "save: $log_file"
