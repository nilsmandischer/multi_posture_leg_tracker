# Position and confidence of a leg 
geometry_msgs/Point position
float32 confidence

# Label: leg or squat or background
uint8 label

uint8 LABEL_BKGD = 0
uint8 LABEL_LEG = 1
uint8 LABEL_SQUAT = 2
