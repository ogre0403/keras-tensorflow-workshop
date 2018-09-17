TAG = $(shell date '+%Y%m%d')

build-dev-img:
	docker build -t ogre0403/keras-tensorflow:$(TAG) .

build-release-img:
	docker build -t nchcai/train:jimmy-nhri-$(TAG) -f Dockerfile.gpu .

run-dev:
	 docker run --rm -it -p 38888:8888 -p 6006:6006  -v `pwd`/:/notebooks ogre0403/keras-tensorflow:$(TAG)  jupyter notebook --allow-root --NotebookApp.token=''

push-img:
	docker push ogre0403/keras-tensorflow:$(TAG)

