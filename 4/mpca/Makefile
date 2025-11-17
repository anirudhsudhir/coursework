build-container-image:
	docker build -t armsim-container-image .

run-container:
	export DISPLAY=host.docker.internal:0
	docker run --rm -e DISPLAY=$$DISPLAY -v ./lab/:/usr/armsim/lab armsim-container-image
