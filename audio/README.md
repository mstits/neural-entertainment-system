# Audio

Each directory here is `<game_name>/` — matching the profile's `name:` field
lowercased. Three tiers of audio sources, checked in order:

1. **NSF (authentic NES APU)** — drop `<game>/<game>.nsf` and we'll play it
   through libgme, the reference NES sound-chip emulator. This is literally
   the same synthesis a real console produces.

   * `brew install game-music-emu` is required.
   * Song-ID → track mapping defaults to `song_id % track_count`. Override
     per-game via `<game>/tracks.map` with lines like `5 12  # overworld`.

2. **Pre-rendered files** — `<game>/<song_id>.ogg` (or `.wav` / `.mp3` /
   `.flac`). Used when NSF isn't present. Song-ID literally names the file.

3. **Procedural synth** — deterministic-per-song-ID fallback. Different
   songs sound different, but nothing sounds like the actual game. This
   is just so something plays when neither of the above is set up.

The mixer watches each game's "current song" RAM byte (see
`src/audio/ram_music.py`'s `SONG_ADDRESSES`) and swaps the active track
whenever it changes.
