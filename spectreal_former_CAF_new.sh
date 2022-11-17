attention_depth=(1 2 4 8 10 12)
model_type="SPECTRAL_FORMER"
refactor_eta=40
mask_ratio=0.8
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
_fifofile="$$.fifo"
mkfifo $_fifofile     # 创建一个FIFO类型的文件
exec 6<>$_fifofile    # 将文件描述符6写入 FIFO 管道， 这里6也可以是其它数字
rm $_fifofile         # 删也可以，

degree=3  # 定义并行度

#根据并行度设置信号个数
#事实上是在fd6中放置了$degree个回车符
for ((i=0;i<${degree};i++));do
    echo
done >&6

for depth in ${attention_depth[@]}; do
    # 从管道中读取（消费掉）一个字符信号
    # 当FD6中没有回车符时，停止，实现并行度控制
    read -u6
    {
        #实际执行的命令
        python houston_program_analyze_depth.py --cfg ${cfg} --batch-size ${batch_size} --data-source-path ${data_source_path} --label-source-path ${label_source_path} --data-target-path ${data_target_path} --label-target-path ${label_target_path} --output ${output} --tag ${tag}_${model_type}_"depth"_${depth} --local_rank ${local_rank} --eta ${eta} --mask-ratio ${mask_ratio} --refactor-eta ${refactor_eta} --attention-depth ${depth}

        echo >&6 # 当进程结束以后，再向管道追加一个信号，保持管道中的信号总数量
    } &
done

wait # 等待所有任务结束

exec 6>&- # 关闭管道
