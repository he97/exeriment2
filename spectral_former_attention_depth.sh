attention_depth=(1 2 4 8 10)
model_type="SPECTRAL_FORMER"
refactor_eta=80
mask_ratio=0.3
#--cfg
cfg="configs/spectral_former/houston_caf.yaml"
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
tag="houston_depth"
#--local_rank
local_rank=1
#eta
eta=0.01
for depth in ${attention_depth[@]}; do
  python houston_program_analyze_depth.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${model_type}_"depth"_${depth} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio} --refactor-eta ${refactor_eta} --attention-depth ${depth}
done