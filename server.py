import socket
import json

def start_udp_server():
    # Set up the UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ("127.0.0.1", 5052)
    server_socket.bind(server_address)

    print(f"Listening on {server_address}")

    while True:
        # Receive data from the client
        data, address = server_socket.recvfrom(1024)  # Buffer size is 1024 bytes
        print(f"Received data from {address}: {data.decode('utf-8')}")
        
        try:
            # Attempt to parse the received data as JSON
            parsed_data = json.loads(data.decode('utf-8'))
            print(f"Parsed JSON data: {parsed_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")

if __name__ == "__main__":
    start_udp_server()
