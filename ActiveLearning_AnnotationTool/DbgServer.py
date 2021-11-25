
import socket


buffer_size  = 4096


udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 10001))


print('* DbgServer online.')

log_file__path = 'dbglog.txt'
log_file_handle = open(log_file__path, 'wb')

while(True):
    message, _ = udp_socket.recvfrom(buffer_size)

    print(message.decode())

    log_file_handle.write(message)
    log_file_handle.write(b'\n')
    log_file_handle.flush()

