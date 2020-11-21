FROM tensorflow/tensorflow:2.2.1-gpu-py3-jupyter

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py


###### customisation from here ########
RUN pip install --user boto3 python-dotenv smart-open shapely scipy
USER root
RUN sudo apt-get update
RUN sudo apt-get install -y openssh-server tmux curl unzip

RUN sudo mkdir /var/run/sshd
RUN sudo echo 'appuser:makefog' | chpasswd
RUN sudo sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sudo sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN sudo echo "export VISIBLE=now" >> /etc/profile


#RUN sudo /usr/bin/ssh-keygen -A
#RUN sudo /usr/bin/ssh-keygen -t rsa -b 4096 -f ssh_host_rsa_key

EXPOSE 22
EXPOSE 6006
EXPOSE 5000
EXPOSE 8888
EXPOSE 8080

#CMD ["/usr/sbin/sshd", "-D"]
#COPY start.sh start.sh
#CMD ["sudo", "service", "ssh", "start"]
#CMD start.sh
#ENTRYPOINT sudo service ssh start && bash
#RUN sudo ssh-keygen -A
#RUN sudo /usr/bin/ssh-keygen -A

RUN sudo ssh-keygen -P "" -t dsa -f /etc/ssh/ssh_host_dsa_key
#RUN sudo ssh-keygen -t rsa -b 4096 -f  /etc/ssh/ssh_host_rsa_key -y
CMD /usr/sbin/sshd -D
#ENTRYPOINT sudo service ssh restart
#USER appuser
#USER appuser
# run it, for example:
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
# Change ownership if necessary
# sudo chown appuser detectron2