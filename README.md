Example commands:

  # Basic usage (uses all defaults)
  uv run python -m rpa.process_runners \
    --input /path/to/video.mp4 \
    --output-dir /path/to/output

  # Custom dynamic ROI settings
  uv run python -m rpa.process_runners \
    --input /Users/nirgofman/Desktop/running-pattern/processed_videos/11RN_RUN2_CAM1/clips/lap_026.mp4 \
    --output-dir ./output_test \
    --roi-height-ratio 0.30 \
    --min-roi-size 150 \
    --height-smoothing-window 20 \
    --min-frames 100

  # With visualization
  uv run python -m rpa.process_runners \
    --input /path/to/video.mp4 \
    --output-dir /path/to/output \
    --visualize


/Users/nirgofman/Desktop/running-pattern/processed_videos/11RN_RUN2_CAM1/clips/lap_026.mp4


TODO:
edge cases in data process pipeline:
- do not accept clips/frames/tracks(runner) where there are another figure behind - (?)
- side camera - /Users/nirgofman/Desktop/running-pattern/processed_videos/18FG_RUN2_CAM2/clips/lap_3.mp4

 uv run python -m rpa.process_runners \
    --input /Users/nirgofman/Desktop/running-pattern/processed_videos/04HN_RUN2_CAM3/clips/lap_006.mp4 \
    --output-dir ./output_test \
    --roi-height-ratio 0.50 \
    --min-roi-size 150 \
    --height-smoothing-window 5 \
    --min-frames 40


# Process all lap clips from a specific camera
for f in /Users/nirgofman/Desktop/running-pattern/tagged_data_videos/11RN_RUN2_CAM2/clips/*.mp4; do
    uv run python -m rpa.process_runners \
        --input "$f" \
        --output-dir /Users/nirgofman/Desktop/running-pattern/rpa/12122025-testings \
        --roi-height-ratio 0.50 \
        --min-roi-size 150 \
        --height-smoothing-window 5 \
        --min-frames 40 \
        --min-speed-ratio 0.5 \
        --top-n-fastest 2 \
        --min-ankle-variance 400
done


video with more people standing to isolate runner: /Users/nirgofman/Desktop/running-pattern/tagged_data_videos/11RN_RUN2_CAM2/clips/lap_006_000.mp4

checkpoint - overview until 21AN_RUN1_CAM1

uv run python -m rpa.batch_process \
    --input-dir /Users/nirgofman/Desktop/running-pattern/tagged_data_videos \
    --output-dir /Users/nirgofman/Desktop/running-pattern/training_clips


  {camera_feature}_{lap_info}_CUT_{clip_num}_{label}.mp4
       │               │           │           │
       │               │           │           └── Original label (000, 001, etc.)
       │               │           └── Clip number from process_runners
       │               └── Lap identifier (lap_017)
       └── Camera/run identifier (15VR_RUN2_CAM2)