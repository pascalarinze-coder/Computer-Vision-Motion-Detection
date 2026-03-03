import cv2
import numpy as np
import time
import numpy as np

#Divide each frame into a 5 × 5 grid
#No noise movement in the region 
#Detected motion area displayed as a live negative (inverted) image 
#neigbouring cells activated contiguously
#resets after continuous 10-second period of no motion detection across 5 x 5 grid


#a function to check the active status of the neighbouring cells. 
def is_neighbor_active(grid, row, col):
    """Checks if any of the 8 neighbors are currently active."""
    # d_row and d_col cover all 8 directions around the cell
    for d_row in [-1, 0, 1]:
        for d_col in [-1, 0, 1]: 
            if d_row == 0 and d_col == 0: # skip the current cell
                continue
            n_row, n_col = row + d_row , col + d_col #neigbour row and col
            if 0 <= n_row < 5 and 0 <= n_col < 5: # Check if neighbor is within the 5x5 grid boundaries
                if grid[n_row, n_col]: # Check if this neighbor was active in the previous frame
                    return True
    return False

# Initialize DroidCam at index 2 
cap = cv2.VideoCapture(2)

#check if camera opened successfully
if not cap.isOpened:
    print('Unable to read camera feed')
    exit(1)

# Variables for motion tracking
active_grid = np.zeros((5, 5), dtype=bool) #2D np table of 5rows & 5cols of initial values 0
any_motion_ever = False
last_motion_time = time.time()

# Background subtraction setup, ditinguishes between static background and moving objects.
background_subtraction = cv2.createBackgroundSubtractorMOG2(history=500,  #number of previous frames  
                                          varThreshold=50, #sensitivity, lower number increases detection sensitivity
                                          detectShadows=False) #pure black and white mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy for processing and display
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    dy, dx = height / 5, width / 5

    # Motion Detection Processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # Noise reduction
    foreground_mask = background_subtraction.apply(gray) # flags where movement is occuring(white pixel) and represent the static background(black pixel) 
    _, thresh = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY) #High contrast black and white map

    motion_detected_this_frame = False

    # Horizontal grid path, the code moves across the cols in each row
    for row in range(5):
        for col in range(5):
            y1, y2 = int(row * dy), int((row + 1) * dy)  
            x1, x2 = int(col * dx), int((col + 1) * dx)
            
            # Extract ROI from threshold to check for motion
            RoI_thresh = thresh[y1:y2, x1:x2] #analyse each cell
            white_count = np.count_nonzero(RoI_thresh) #counts white pixels (motion)

            # Only trigger if motion is significant (ignore noise)
            if white_count > 1000: 
                # Spreading logic (First trigger OR neighbor is active)
                if not any_motion_ever or is_neighbor_active(active_grid, row, col):
                    active_grid[row, col] = True
                    any_motion_ever = True
                    motion_detected_this_frame = True
                    last_motion_time = time.time()

            # Live Negative (Inverted) Image for active regions
            if active_grid[row, col]:
                display_frame[y1:y2, x1:x2] = 255 - frame[y1:y2, x1:x2]
            else:
                # Keep the original "normal" camera view for this ROI
                display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # 10-second Reset
    if time.time() - last_motion_time > 10:
        active_grid.fill(False)
        any_motion_ever = False

    # Draw the Grid Lines (for visualization)
    for i in range(1, 5):
        cv2.line(display_frame, (int(i * dx), 0), (int(i * dx), height), (0, 255, 0), 1)
        cv2.line(display_frame, (0, int(i * dy)), (width, int(i * dy)), (0, 255, 0), 1)

    cv2.imshow('Motion Mask', thresh)
    cv2.imshow('Motion Detection Reality Model', display_frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
