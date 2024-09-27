executable = MassTest.sh
arguments = $(test)
getenv = true
error = logs/$(test).error
output = logs/$(test).out
log = logs/$(test).log
notification = never
request_memory = 32 GB

queue test from tests.txt
