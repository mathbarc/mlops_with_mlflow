all: build

build:
	docker build ./sample_api/ -t docker-registry.solv.local/rice_classifier
	docker push docker-registry.solv.local/rice_classifier

