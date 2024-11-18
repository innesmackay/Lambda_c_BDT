executable = Apply.sh
arguments = $(config)
getenv = true
error = logs/Apply_$(config).error
output = logs/Apply_$(config).out
log = logs/Apply_$(config).log
notification = never
request_memory = 32 GB
queue 1

queue config from configs.txt
