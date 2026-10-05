Lucy mascot pose atlas

The app expects the extracted mascot atlas at:
  mobile/app/src/main/assets/lucy_pose_atlas.webp

The atlas is a 528x771 WebP containing 67 cropped Lucy poses. LucyApp.kt uses fixed source rectangles to render one pose at a time, so the full sheet is never displayed.

Pose selection is state-driven:
- idle: default
- typing: user is entering text
- listening: microphone button
- processing: message is being submitted
- success: task submission completed
- plus a tap-to-cycle demo through several expressive poses

The atlas is intentionally kept as one binary asset to avoid 67 separate Android resources.
