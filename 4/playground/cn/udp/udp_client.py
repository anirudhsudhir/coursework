import socket
import sys

if len(sys.argv) < 3:
    print("Usage: python udp_client.py [server_addr] [server_port]")
    sys.exit(1)

client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_sock.settimeout(1)
print('Starting UDP Client: (Type "!quit!" to quit the client)')
while True:
    msg = input("Enter a message: ")
    if not msg or msg == "!quit!":
        client_sock.close()
        break
    client_sock.sendto(msg.encode(), (sys.argv[1], int(sys.argv[2])))
    print("Response from server:")
    resp = client_sock.recv(1024)
    print(resp.decode())
