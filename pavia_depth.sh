#etas=(0.05 0.1 0.2 0.4 0.5 0.75 0.9 1.0)
#0.4:5,0.5:6,0.6:8,0.7:9,0.8:10,0.9:11
#refactor_etas=(0.0 10 20 40 80 160)
#refactor_etas=(100 120 140)
#refactor_etas=(60 500 1000 2000 5000 10000)
#attention_depth=(1 2 4 8 10 12 14 16)
attention_depth=(8 10 12)
model_type="pavia"
refactor_eta=40
mask_ratio=0.6
#--cfg
cfg="configs/Dtransformer/pavia.yaml"
#--batch-size
batch_size="32"
#--data-source-path
data_source_path="dataset/pavia-7/UP/paviaU.mat"
#--label-source-path
label_source_path="dataset/pavia-7/UP/paviaU_gt_7.mat"
#--data-target-path
data_target_path="dataset/pavia-7/PC/pavia.mat"
#--label-target-path
label_target_path="dataset/pavia-7/PC/pavia_gt_7.mat"
#--output
output="outputs"
#--tag
tag="pavia"
#--local_rank  python houston_program_analyze_depth.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${model_type}_"depth"_${depth} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio} --refactor-eta ${refactor_eta} --attention_depth ${depth}

local_rank=0
#eta
eta=0.01
for depth in ${attention_depth[@]}; do
  python pavia.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${model_type}_"depth"_${depth} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio} --refactor-eta ${refactor_eta} --attention-depth ${depth}
done