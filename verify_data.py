# from data.loader import ParkingProtocol, Sequence
from data import ParkingProtocol, Sequence

def main():
    DATA_ROOT = "C:/Users/vikas/Downloads/DJI_0001"
    
    # Init in TRAJECTORY mode
    dataset = ParkingProtocol(DATA_ROOT, mode='trajectory')
    
    if len(dataset) == 0:
        print("No sequences found.")
        return

    # Pick the first sequence
    seq: Sequence = dataset[0]
    
    print(f"\n--- Checking Sequence ---")
    print(f"Sequence ID: {seq.seq_id}")
    print(f"Length: {len(seq)} frames")
    
    # Check first and last frame to ensure temporal order
    first_frame = seq[0]
    last_frame = seq[-1]
    
    print(f"Start Frame: {first_frame.frame_idx} (File: {first_frame.frame_id})")
    print(f"End Frame:   {last_frame.frame_idx} (File: {last_frame.frame_id})")
    
    if last_frame.frame_idx > first_frame.frame_idx:
        print("✅ Temporal Order Verified.")
    else:
        print("❌ Temporal Order Incorrect!")

if __name__ == "__main__":
    main()