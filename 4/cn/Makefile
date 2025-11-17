build-container-image:
	docker build --no-cache --platform=linux/amd64 -t ns2-container-image .

run-container:
	# export DISPLAY=host.docker.internal:0
	docker run -e DISPLAY=$$DISPLAY --rm -it -v ./lab-files:/usr/cn/lab-files ns2-container-image

nsg2:
	java -jar ./resources/NSG2.1.jar
