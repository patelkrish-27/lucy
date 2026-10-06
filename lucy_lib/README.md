# lucy_lib

The approved Lucy pose library — the canonical store the TUI reads from at
runtime (`crates/lucy-mascot` loads these PNGs from disk first, with
`LUCY_POSE_DIR` as an override, and only falls back to its embedded copies).

## Current set: hand-cropped poses (`ref/Lucy_mascot/`)

45 poses, cleaned from the black-background crops (background removed,
transparent PNG, tight crop, native resolution — no upscale):

- `standing_idle`, `winking_noticed`, `laughing_hearts`, `star_celebration`,
  `listening_wave`, `listening_curious`, `questioning`, `arms_crossed`,
  `laughing_waves`, `pompoms_cheer`, `sweet_sparkle`, `hugging_heart`,
  `laughing_confetti`, `greeting_sparkle`, `sunny_laugh`, `singing_melody`,
  `laptop_heart`, `typing_angry`, `laptop_music`, `laptop_angry`,
  `writing_clipboard`, `idea_bulb`, `reading_book`, `laptop_frustrated`,
  `laptop_smile`, `headphones_closed`, `singing_mic`, `headphones_sway`,
  `humming_wave`, `headphones_rest`, `sleeping_belly`, `sleeping_curled`,
  `morning_sun`, `giggle_cheeks`, `crying_tears`, `crying_puddle`,
  `steam_angry`, `yawning`, `dizzy_swirl`, `bored`, `skeptical`,
  `angry_crossvein`, `annoyed_red`, `playful_wink`, `worried_hands`.

The 7 moods the TUI shows map one-to-one onto these poses
(`pose_for_mood` in `crates/lucy-mascot/src/lib.rs`):

- Idle → `standing_idle`, Listening → `headphones_closed`,
  Thinking → `questioning`, Working → `laptop_smile`,
  Talking → `singing_melody`, Happy → `star_celebration`,
  Approval → `sunny_laugh`.

To swap art, replace the PNG in `poses/` keeping the same name and restart
`lucy` — no rebuild needed.

## Retained older sheet set

The earlier `ref/lucy_mascot_library_withboundary.png` extraction is kept
alongside: the `cNN` slices plus the old semantic names (`idle`, `happy`,
`listening`, `thinking`, `working`, `talking`, `approval`). They are not
used by the TUI anymore, only archived here.
