etas=(0.01 0.03 0.05 0.1 0.2 0.4 0.5 0.75 0.9 1.0)
#--cfg
cfg="configs/Dtransformer/houston_eta_maskrt_refactor.yaml"
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
tag="eta"
#--local_rank
local_rank=0
#mask ratio
mask_ratio=0.6
refactor_eta=1
depth=2
for eta in ${etas[@]}; do
  python houston_program_analyze_depth.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${model_type}_"eta"_${eta} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio} --refactor-eta ${refactor_eta} --attention-depth ${depth}
done