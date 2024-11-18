executable = TrainAndApply.sh
arguments = $(config)
getenv = true
error = logs/TrainAndApply_$(config).error
output = logs/TrainAndApply_$(config).out
log = logs/TrainAndApply_$(config).log
notification = never
request_memory = 32 GB
queue 1

queue config from configs.txt
