# One piece of litter, as reported by perception.

# Lifecycle states (set by the object registry / state machine).
uint8 STATE_SEEN=0        # detected, not yet confirmed across frames
uint8 STATE_CONFIRMED=1   # seen enough times to act on
uint8 STATE_TARGETED=2    # the robot is currently going after it
uint8 STATE_PICKED=3      # successfully grasped and removed
uint8 STATE_FAILED=4      # grasp attempts exhausted

uint32 id                     # stable across updates of the same physical object
string label                  # bottle | can | cup | paper | other
float32 label_confidence      # classifier confidence, 0..1
float32 detection_confidence  # detector confidence, 0..1
string bin                    # where it goes: recycling | landfill (see perception config/bins.yaml)

# Object center on the floor. header.frame_id is normally "map" so the
# position stays valid while the robot moves; transform with tf2 as needed.
geometry_msgs/PointStamped position

float32 width   # meters, largest horizontal extent (use for gripper aperture)
float32 height  # meters, above the floor

bool depth_valid  # false if position came from the floor-plane fallback
uint8 state
