FROM tensorflow/tensorflow:1.9.0


RUN pip install keras

RUN rm /notebooks/*

ADD . /notebooks

EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root", "--NotebookApp.token=''"]
