import socket
import sys
import signal

if len(sys.argv) < 2:
    print("Usage: python udp_server.py [server_port]")
    sys.exit(1)

server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_sock.bind(("", int(sys.argv[1])))


def sig_handler(signum, frame):
    server_sock.close()
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)

print("UDP server is listening")
while True:
    (msg, client_address) = server_sock.recvfrom(1024)
    data = msg.decode()
    print(f"Received data from client: {data}")
    data = data.upper()
    server_sock.sendto(data.encode(), client_address)
