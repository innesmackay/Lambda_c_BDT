executable = Train.sh
arguments = $(config)
getenv = true
error = logs/Train_$(config).error
output = logs/Train_$(config).out
log = logs/Train_$(config).log
notification = never
request_memory = 32 GB
queue 1

queue config from configs.txt
