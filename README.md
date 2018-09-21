# keras-tensorflow-workshop

## How to use this repository in your home

Step 1: Install Docker 

[for Mac](https://docs.docker.com/docker-for-mac/install/)

[For Linux](https://docs.docker.com/install/linux/docker-ce/centos/)

[for Windows](https://docs.docker.com/docker-for-windows/install/)

Step 2: Build image

```sh
$ docker build -t ogre0403/keras-tensorflow:latest .
```

Step 3: Run lab environment

```sh
$ docker run --rm -it -p 8888:8888 -v `pwd`/:/notebooks ogre0403/keras-tensorflow:latest  jupyter notebook --allow-root --NotebookApp.token=''
```
