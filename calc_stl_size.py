"""Calculate STL file sizes for different triangle counts."""

print('Format STL Binary:')
print('  - 80 bytes header')
print('  - 4 bytes triangle count')
print('  - 50 bytes per triangle')
print('    * 12 bytes normal (3 × float32)')
print('    * 36 bytes vertices (3 × 3 × float32)')
print('    * 2 bytes attribute')
print()

print('Twój aktualny plik:')
triangles = 791_200
size_mb = 37.73
theoretical = (84 + triangles * 50) / 1024 / 1024
print(f'  Trójkąty: {triangles:,}')
print(f'  Rozmiar rzeczywisty: {size_mb:.2f} MB')
print(f'  Teoretyczny: {theoretical:.2f} MB')
print()

print('Rozmiary dla różnych ilości trójkątów:')
print('-' * 50)
for triangles in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]:
    size_mb = (84 + triangles * 50) / 1024 / 1024
    if size_mb < 1024:
        print(f'  {triangles:>12,} trójkątów → {size_mb:>8.2f} MB')
    else:
        print(f'  {triangles:>12,} trójkątów → {size_mb/1024:>8.2f} GB')

print()
print('Dla 20 mln trójkątów: ~954 MB (~0.93 GB)')
