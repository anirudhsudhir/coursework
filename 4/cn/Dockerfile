FROM ubuntu:latest

WORKDIR /usr/cn
COPY ./resources/nam_1.14_amd64.deb .

RUN apt update -y && apt upgrade -y
RUN apt install ns2 -y
RUN apt install vim tcl -y
RUN apt install default-jre -y --fix-missing
RUN dpkg -i nam_1.14_amd64.deb

CMD ["bash"]
