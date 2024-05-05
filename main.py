from utils import read_video, save_video
from trackers import Tracker
from player_clustering import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance import SpeedDistanceEstimator

import cv2
import numpy as np

def main():
    # read video
    video_frames = read_video('input_videos/08fd33_4.mp4', max_frames=100)
    print('--- Frames extraction is done ----')

    # initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_trackers(video_frames,read_from_stub=True, stub_path='stubs/track_stubs_100frames.pkl')
    # get object positions
    tracker.add_position_to_tracks(tracks)


    # #save cropped images of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save the cropped image
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
    #     break





    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub_100f.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    # speed and distance estimator
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],  track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    # assign the ball to the player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)





    # draw object tracks
    output_video_frames = tracker.draw_annotation(video_frames, tracks, team_ball_control) 

    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # draw spped distance
    speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # save video
    save_video(output_video_frames, 'output_videos/video_100frames.avi', frame_rate=24, resize=None, codec='XVID')
    print('---- Saved final output successfully -----')


if __name__=='__main__':
    main()