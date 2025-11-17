FROM ubuntu:latest

WORKDIR /usr/armsim

COPY ./armsim_linux.zip .
RUN apt update -y && apt upgrade -y
RUN apt install mono-complete -y
RUN apt install unzip -y

RUN unzip armsim_linux.zip

CMD ["mono","armsim/ARMSim.exe"]
