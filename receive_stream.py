import socket
import cv2
import numpy as np

# Configuration
LISTEN_IP = "0.0.0.0" # Listen on all interfaces
LISTEN_PORT = 5052    # Must match send_frame port
MAX_DGRAM = 65536     # Max buffer size for recv

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    
    print(f"Listening on {LISTEN_IP}:{LISTEN_PORT}...")
    print("Press 'q' or ESC to exit.")

    buffer = b""
    expected_size = 0
    
    try:
        while True:
            # Receive packet
            data, addr = sock.recvfrom(MAX_DGRAM)
            
            # Simple protocol logic:
            # If we are waiting for a new frame (buffer empty), 
            # we expect the first packet to be the 4-byte size header.
            # (Note: This is brittle over UDP if packets reorder/drop, 
            # but matches the provided sender logic).
            
            if expected_size == 0:
                # We are looking for a header
                if len(data) == 4:
                    expected_size = int.from_bytes(data, 'big')
                    buffer = b""
                    # print(f"New frame size: {expected_size}") # Debug
                else:
                    # Received data when expecting header? 
                    # Drop it or append if we assume we missed the header?
                    # For safety, drop and wait for next clean header (size 4)
                    continue
            else:
                # We are collecting frame data
                buffer += data
                
                if len(buffer) >= expected_size:
                    # Frame complete
                    frame_data = buffer[:expected_size]
                    
                    # Reset for next frame immediately to avoid buffering old data
                    # (Handling the leftover in buffer if any, though UDP boundaries usually separate logic)
                    # In this simple protocol, sender sends discrete packets. 
                    # We just assume 'buffer' exactly matches expected_size or is slightly more 
                    # if we messed up, but let's stick to strict clearing.
                    buffer = b"" 
                    expected_size = 0

                    # Decode
                    frame_arr = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        cv2.imshow("RPi Stream Configured", frame)
                    else:
                        print("Failed to decode frame")

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("Stream stopped.")

if __name__ == "__main__":
    main()
