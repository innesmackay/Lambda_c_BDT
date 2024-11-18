executable = snake.sh
arguments = "$(config) $(rule)"
getenv = true
error = logs/snake_$(config)_$(rule).error
output = logs/snake_$(config)_$(rule).out
log = logs/snake_$(config)_$(rule).log
notification = never
request_memory = 32 GB
queue config,rule from configs.txt
