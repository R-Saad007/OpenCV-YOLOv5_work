# Scripts:
Includes scripts for different purposes.

# Logic (handler.py):
We need to implement logic to count footfall for a given video clip.

## Simple Logic:
- Check for track ID of an object and increment whenever the object crosses the designated area (green area) **FROM ABOVE** (Ensure only movement towards the area from top to bottom is considered)
- Making use of 2 dictionaries to store the state of the tracked object (inside,outside)

## Cases Handled:
- People entering
- People re-entering(same individual have similar track_id)
- Ignore objects leaving or already inside (need to think of logic)
