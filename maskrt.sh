#etas=(0.05 0.1 0.2 0.4 0.5 0.75 0.9 1.0)
#0.4:5,0.5:6,0.6:8,0.7:9,0.8:10,0.9:11
mask_ratios=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#--cfg
cfg="configs/Dtransformer/houston_analyze_maskrt.yaml"
#--batch-size
batch_size="32"
#--data-source-path
data_source_path="dataset/houston13-18/Houston13.mat"
#--label-source-path
label_source_path="dataset/houston13-18/Houston13_7gt.mat"
#--data-target-path
data_target_path="dataset/houston13-18/Houston18.mat"
#--label-target-path
label_target_path="dataset/houston13-18/Houston18_7gt.mat"
#--output
output="outputs"
#--tag
tag="houston_1G_2C_1Decoder_10seed_maskrt"
#--local_rank
local_rank=0
#eta
eta=0.01
for mask_ratio in ${mask_ratios[@]}; do
  python houston_program_analyze_maskrt.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${mask_ratio} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio}
done