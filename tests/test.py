from chart_parser2 import load_chart

song_path = "songs/LL COOL J - Headsprung (MrDrummerboy97)/notes.chart"

song = load_chart(
    song_path,
    difficulty="Expert"
)

print(f"Loading song from: {song_path}")
print("Loaded song with", len(song), "notes.")
print("First 5 notes:")
for note in song[:5]:
    print(note)

# env = GuitarHeroEnv(
#     song,
#     dt=0.02,
#     hit_window=0.10,
#     lookahead=10,
#     render_mode="rgb_array"
# )


# obs, info = env.reset()

# # Structured chart observation
# print(obs["upcoming_notes"])

# # Image observation
# screen = obs["screen"]

# print(screen.shape)
# (96, 64, 3)