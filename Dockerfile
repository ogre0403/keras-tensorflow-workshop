FROM tensorflow/tensorflow:1.9.0


RUN pip install keras jupyterthemes

RUN jt -t oceans16


RUN rm /notebooks/*

ADD . /notebooks

EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root", "--NotebookApp.token=''"]
