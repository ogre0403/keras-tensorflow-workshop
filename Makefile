TAG = $(shell date '+%Y%m%d')

build-img:
	docker build -t ogre0403/keras-tensorflow:$(TAG) .

run-dev:
	 docker run --rm -it -p 8888:8888 -p 6006:6006  -v `pwd`/:/notebooks  tensorflow/tensorflow:1.9.0 jupyter notebook --allow-root --NotebookApp.token=''	

push-img:
	docker push ogre0403/keras-tensorflow:$(TAG)
