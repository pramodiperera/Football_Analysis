import cv2

# def read_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         flag, frame = cap.read()
#         if not flag:
#             break
#         frames.append(frame)
#     return frames

def read_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        frames.append(frame)
        
        if max_frames is not None and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

# def save_video(output_video_frames, output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],output_video_frames[0].shape[1]) )
#     for frame in output_video_frames:
#         out.write(frame)
#     out.release()

def save_video(output_video_frames, output_video_path, frame_rate=24, resize=None, codec='XVID'):
    if resize:
        output_video_frames = [cv2.resize(frame, resize) for frame in output_video_frames]

    height, width, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
